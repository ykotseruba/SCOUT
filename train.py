import argparse
import glob, os
import datetime
import torch
import sys
import time
import torch.nn as nn
import pickle as pkl
from torch.autograd import Variable
from torchvision import transforms, utils
from PIL import Image
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F
from dataloader import DReyeVEDataset, BDDADataset
from loss import cc, sim, nss, kldiv
import cv2
import model
from utils import img_save, AverageMeter, num_params, loss_func
from tqdm import tqdm
import yaml
import shutil
import copy
import bz2
import torch.optim.lr_scheduler as lr_scheduler
import pandas as pd
from os.path import join
from utils import get_task_attribute_dict

from torchsummary import summary

blur_func = transforms.GaussianBlur(11, 2)

def get_model_class(model_class_name):

    obj = getattr(model, model_class_name, None)
    if obj is None:
        raise ValueError(f'Error: model {model_class_name} is not supported!')
    else:
        return obj


class Train:
    def __init__(self, config_file='configs/default.yaml', save_dir=None):
        self.configs = self.load_config(config_file)

        self.dataset_name = self.configs['dataset']
        if self.dataset_name == 'DReyeVE':
            dataset_class = DReyeVEDataset
        elif self.dataset_name == 'BDD-A':
            dataset_class = BDDADataset

        self.model_params = self.configs['model_params']
        self.task_attributes = get_task_attribute_dict(self.model_params.get('task_attributes', None))
        self.map_params = self.model_params.get('map_params', None)
        self.train_params = self.configs['train_params']
        self.test_params = self.configs['test_params']

        self.use_task = self.model_params.get('use_task', False)
        self.use_map = self.model_params.get('use_map', False)

        if self.use_task and self.use_map:
            raise ValueError('ERROR: cannot use both task and map. \
                             Set "use_task" or "use_map" to False in config.')

        self.save_dir = save_dir
        self.results_dir = None
        self.save_config = None
        self.setup_save_dir(config_file)

        print(f'-> Building {self.configs["model_class"]}')
        self.model = get_model_class(self.configs['model_class'])(**self.model_params)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.device_count() > 1:
            print('-> Detected', torch.cuda.device_count(), 'GPUs')
            self.model = nn.DataParallel(self.model)

        self.model.to(self.device)

        params = list(filter(lambda p: p.requires_grad, self.model.parameters())) 
        
        if self.train_params['optimizer'] == 'Adam':
            self.optimizer = torch.optim.Adam(params, lr=self.train_params['lr'])
        elif self.train_params['optimizer'] == 'AdamW':
            self.optimizer = torch.optim.AdamW(params, lr=self.train_params['lr'])
        else:
            raise ValueError(f'ERROR: optimizer {self.train_params["optimizer"]} is not supported!')

        if self.train_params['lr_sched']:
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=2)


        np.random.seed(0)
        torch.manual_seed(0)

        

        train_dataset = dataset_class(self.model_params['clip_size'],
                                       old_gt=False,
                                       mode='train',
                                       weight_type=self.train_params['weight_type'],
                                       img_size=self.model_params['img_size'],
                                       task_attributes=self.task_attributes,
                                       map_params=self.map_params)
        train_dataset.setup()

        self.train_sample_weights = train_dataset.get_sample_weights()        

        if self.train_params['weighted_sampler']:
            weighted_sampler = torch.utils.data.WeightedRandomSampler(self.train_sample_weights,
                                                                      len(train_dataset), 
                                                                      replacement=True)
        else:
            weighted_sampler = None
        
        val_dataset = dataset_class(self.model_params['clip_size'],
                                     old_gt=False,
                                     mode='val',
                                     weight_type=self.train_params['weight_type'],
                                     img_size=self.model_params['img_size'],
                                     task_attributes=self.task_attributes,
                                     map_params=self.map_params)
        val_dataset.setup()

        self.train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                        batch_size=self.train_params['batch_size'],
                                                        shuffle=not self.train_params['weighted_sampler'], 
                                                        sampler=weighted_sampler,
                                                        num_workers=self.train_params['no_workers'])
        self.val_loader = torch.utils.data.DataLoader(val_dataset, 
                                                        batch_size=self.train_params['batch_size'], 
                                                        shuffle=False, 
                                                        num_workers=self.train_params['no_workers'])

        num_params(self.model)

    def setup_save_dir(self, config_file):
        if self.save_dir is None:
            datestring = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            self.save_dir = f'train_runs/{datestring}'
        
        os.makedirs(self.save_dir, exist_ok=True)

        self.results_dir = os.path.join(self.save_dir, 'results')
        os.makedirs(self.results_dir, exist_ok=True)

        self.save_config = f'{self.save_dir}/config.yaml'
        print(f'-> Saving src/configs to train_runs/{datestring}...')
        src_dir = os.path.join(self.save_dir, 'src')
        os.makedirs(src_dir, exist_ok=True)
        files = [x for x in os.listdir('.') if x.endswith('.py')]
        for file in files:
            shutil.copy(file, os.path.join(src_dir, file))
        shutil.copy(config_file, self.save_config)


    def load_config(self, config_file):
        with open(config_file, 'r') as fid:
            configs = yaml.safe_load(fid)

        return configs

    def train_epoch(self, epoch):

        self.model.train()
        
        total_loss = AverageMeter()
        cur_loss = AverageMeter()

        num_samples = len(self.train_loader)

        log_interval = self.train_params['log_interval']

        with tqdm(self.train_loader, unit='batch', desc=f'Train epoch {epoch}/{self.train_params["no_epochs"]}') as tepoch:
            for idx, sample in enumerate(tepoch):
                img_clips = sample[0]
                gt_sal = sample[1]
                task_dict_ = sample[2]
                route_map = sample[3]
                vid_ids, frame_ids, sample_idx = [x.tolist() for x in sample[4]]

                #print('sample size', img_clips.shape)
                if self.use_task:
                    task = {}
                    for k, v in task_dict_.items():
                        task[k] = v.to(self.device)
                        #print(k, v.shape)

                if self.use_map:
                    route_map = route_map.to(self.device)

                img_clips = img_clips.to(self.device)
                img_clips = img_clips.permute((0,2,1,3,4))
                gt_sal = gt_sal.to(self.device)
                
                self.optimizer.zero_grad()

                if self.use_task:
                    pred_sal = self.model(img_clips, task)
                elif self.use_map:
                    pred_sal = self.model(img_clips, route_map)
                else:
                    pred_sal = self.model(img_clips, None)
                
                pred_sal = blur_func(pred_sal)
                pred_sal = transforms.Resize(gt_sal.shape[-2:])(pred_sal)

                assert pred_sal.size() == gt_sal.size()

                if self.train_params['weighted_loss']:
                    weights = [self.train_sample_weights[x] for x in sample_idx]
                else:
                    weights = [1]*pred_sal.shape[0]

                weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
                loss = loss_func(pred_sal, gt_sal, weights, self.train_params)
    
                loss.backward()
                self.optimizer.step()
                total_loss.update(loss.item())
                cur_loss.update(loss.item())

                if idx%log_interval==(log_interval-1):
                    tepoch.set_postfix_str(f'Loss: {cur_loss.avg:0.3f}')
                    cur_loss.reset()
                
        print('Train epoch {:2d} avg_loss : {:.5f}'.format(epoch, total_loss.avg))
        sys.stdout.flush()

        return total_loss.avg

    def val_epoch(self, epoch):
        self.model.eval()

        total_loss = AverageMeter()
        total_cc_loss = AverageMeter()
        total_sim_loss = AverageMeter()

        num_samples = len(self.val_loader)
        
        with tqdm(self.val_loader, unit='batch', desc=f'Val epoch {epoch}') as vepoch:
            for idx, sample in enumerate(vepoch):
                img_clips = sample[0]
                gt_sal = sample[1]
                task_dict_ = sample[2]
                route_map = sample[3]

                if self.use_task:
                    task = {}
                    for k, v in task_dict_.items():
                        task[k] = v.to(self.device)

                img_clips = img_clips.to(self.device)
                img_clips = img_clips.permute((0,2,1,3,4))
                gt_sal = gt_sal.to(self.device)
                if self.use_map:
                    route_map = route_map.to(self.device)
                
                self.optimizer.zero_grad()

                if self.use_task:
                    pred_sal = self.model(img_clips, task)
                elif self.use_map:
                    pred_sal = self.model(img_clips, route_map)
                else:
                    pred_sal = self.model(img_clips, None)
                        

                pred_sal = blur_func(pred_sal)
                pred_sal = transforms.Resize(gt_sal.shape[-2:])(pred_sal)

                assert pred_sal.size() == gt_sal.size()

                # compute unweighted loss
                weights = [1]*pred_sal.shape[0]
                weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

                loss = torch.mean(loss_func(pred_sal, gt_sal, weights, self.train_params))
                cc_loss = torch.mean(cc(pred_sal, gt_sal))
                sim_loss = torch.mean(sim(pred_sal, gt_sal))

                total_loss.update(loss.item())
                total_cc_loss.update(cc_loss.item())
                total_sim_loss.update(sim_loss.item())

                if np.isnan(total_cc_loss.avg) or np.isnan(total_sim_loss.avg):
                    break

        print(f'Val epoch {epoch:2d} avg_loss: {total_loss.avg:.5f} cc_loss: {total_cc_loss.avg} sim_loss: {total_sim_loss.avg}')
        sys.stdout.flush()

        return total_loss.avg, total_cc_loss.avg, total_sim_loss.avg


    def train(self):
        best_loss = np.inf
        best_model = None
        best_model_save_name = None
        best_epoch = -1
        early_stop_thresh = 2

        for epoch in range(self.train_params['no_epochs']):
            loss = self.train_epoch(epoch)
            
            with torch.no_grad():
                val_loss, cc_loss, sim_loss = self.val_epoch(epoch)
                if np.isnan(cc_loss) or np.isnan(sim_loss):
                    break
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_epoch = epoch
                    best_model = copy.deepcopy(self.model)
                    model_save_name=f'checkpoint_{epoch}.pt'
                    best_model_save_name = model_save_name
                    print(f'-> Saved {model_save_name}')
                    if torch.cuda.device_count() > 1:    
                        torch.save(self.model.module.state_dict(), os.path.join(self.save_dir, model_save_name))
                    else:
                        torch.save(self.model.state_dict(), os.path.join(self.save_dir, model_save_name))

            if self.train_params['lr_sched']:
                self.scheduler.step(val_loss)
                #print('Current lr=', self.scheduler.get_last_lr())

            if self.train_params['early_stop']:
                if epoch - best_epoch > early_stop_thresh:
                    print(f'Early stop training at epoch {epoch}')
                    break

        print(f'-> Updating {self.save_config}...', end='', flush=True)

        self.configs['best_weights'] = best_model_save_name

        with open(self.save_config, 'w') as fid:
            yaml.dump(self.configs, fid)
        print('done')

        return best_loss, best_epoch, self.save_dir


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml', type=str, help='Path to config file in YAML format')
    parser.add_argument('--save_dir', type=str, default=None, help='Save dir')
    args = parser.parse_args()
    print(args)

    train = Train(args.config, args.save_dir)
    train.train()


