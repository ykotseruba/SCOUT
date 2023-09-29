import argparse
import yaml
import os
from os.path import join
import torch
import pickle as pkl
import pandas as pd
from torchvision import transforms, utils
from tqdm import tqdm
from dataloader import BDDADataset
from dataloader import DReyeVEDataset
from train import get_model_class
from data_utils.dreyeve_data_utils import DReyeVEUtils
from data_utils.bdda_data_utils import BDDAUtils
from utils import get_task_attribute_dict, img_save
from loss import cc, sim, nss, kldiv


blur_func = transforms.GaussianBlur(11, 2)

class Test():
	def __init__(self):
		self.configs = None
		self.train_params = None
		self.test_params = None
		self.model_params = None
		self.results_dir = None
		self.device = None

	def init(self, configs, config_dir, device):
		self.configs = configs
		self.config_dir = config_dir
		self.results_dir = os.path.join(config_dir, 'results')
		self.cache_dir = os.path.join(config_dir, 'cache')
		self.train_params = configs['train_params']
		self.model_params = configs['model_params']
		self.use_task = self.model_params.get('use_task', False)
		self.use_map = self.model_params.get('use_map', False)

		self.dataset_name = self.configs['dataset']

		if self.use_task and self.use_map:
			raise ValueError('ERROR: cannot use both task and map. \
							 Set "use_task" or "use_map" to False in config.')

		self.task_attributes = get_task_attribute_dict(self.model_params.get('task_attributes', None))
		self.map_params = self.model_params.get('map_params', None)
		self.test_params = configs['test_params']
		self.results_dir = os.path.join(config_dir, 'results')
		self.cache_dir = os.path.join(config_dir, 'cache')
		self.device = device
		self.cache_fname = os.path.join(self.cache_dir, 'eval_dict_'+self.configs.get('best_weights', 'checkpoint_0.pt').replace('.pt', '.pkl'))


	def load_saved(self, config_dir):
		print('-> Loading configs... ', end='')
		config_path = f'{config_dir}/config.yaml'
		with open(config_path, 'r') as fid:
			configs = yaml.safe_load(fid)
		self.configs = configs
		self.config_dir = config_dir

		self.dataset_name = self.configs['dataset']

		if self.dataset_name == 'DReyeVE':
			self.datautils = DReyeVEUtils()

		elif self.dataset_name == 'BDD-A':
			self.datautils = BDDAUtils()
		else:
			raise ValueError(f'Dataset {self.dataset_name} is not supported!')

		self.train_params = configs['train_params']
		self.model_params = configs['model_params']
		self.use_task = self.model_params.get('use_task', False)
		self.use_map = self.model_params.get('use_map', False)

		if self.configs['best_weights'] is None:
			print('ERROR: best_weights not found!')
			return False

		if self.use_task and self.use_map:
			raise ValueError('ERROR: cannot use both task and map. \
							 Set "use_task" or "use_map" to False in config.')

		self.task_attributes = get_task_attribute_dict(self.model_params.get('task_attributes', None))
		self.map_params = self.model_params.get('map_params', None)
		self.test_params = configs['test_params']
		self.results_dir = os.path.join(config_dir, 'results')
		self.cache_dir = os.path.join(config_dir, 'cache')
		if self.configs['best_weights'] is not None:
			self.cache_fname = os.path.join(self.cache_dir, 'eval_dict_'+self.configs.get('best_weights').replace('.pt', '.pkl'))
	
		print('done')

		print('-> Loading model... ', end='', flush=True)
		model = get_model_class(self.configs['model_class'])(**self.model_params)
			
		best_model_weights = os.path.join(config_dir, self.configs['best_weights'])
		model.load_state_dict(torch.load(best_model_weights))
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model = model.to(self.device)
		torch.backends.cudnn.benchmark = False
		print('done')
		return True

	def test_saved(self, config_dir, evaluate=True, save_images=False):
		
		if not self.load_saved(config_dir):
			return

		self.test_params['evaluate'] = evaluate
		self.test_params['save_images'] = save_images

		if self.dataset_name == 'DReyeVE':

			for test_vid_id in range(38, 75):
				test_dataset = DReyeVEDataset(self.model_params['clip_size'],
											  old_gt=False, mode='test',
											  img_size=self.model_params['img_size'],
											  task_attributes=self.task_attributes,
											  map_params=self.map_params)
				test_dataset.setup(test_vid_id=test_vid_id)
				test_loader = torch.utils.data.DataLoader(test_dataset, 
															batch_size=self.train_params['batch_size'], 
															shuffle=False, 
															num_workers=self.train_params['no_workers'])    

				self.test(self.model, self.device, test_loader, test_vid_id=test_vid_id)

		
			if self.test_params['evaluate']:
				self.write_results(self.get_eval_dict())

		else:

			test_dataset = BDDADataset(self.model_params['clip_size'],
										   mode='test',
										   weight_type=self.train_params['weight_type'],
										   img_size=self.model_params['img_size'],
										   task_attributes=self.task_attributes,
										   map_params=self.map_params)

			test_dataset.setup()

			test_loader = torch.utils.data.DataLoader(test_dataset, 
														batch_size=self.train_params['batch_size'], 
														shuffle=False, 
														num_workers=self.train_params['no_workers'])    

			self.test(self.model, self.device, test_loader)

		
			if self.test_params['evaluate']:
				self.write_results(self.get_eval_dict())


	def get_eval_dict(self):
		os.makedirs(self.cache_dir, exist_ok=True)
		eval_dict = self.datautils.get_eval_dict(self.cache_fname)

		return eval_dict

	def save_eval_dict(self, eval_dict):
		with open(self.cache_fname, 'wb') as fid:
			pkl.dump(eval_dict, fid)

	def test(self, model, device, test_loader, test_vid_id='all'):
		
		if self.test_params['evaluate']:
			eval_dict = self.get_eval_dict()

		model.eval()
		num_samples = len(test_loader)
		for idx, sample in enumerate(tqdm(test_loader, unit='batch', desc=f'Test vid {test_vid_id}', total=num_samples)):

			img_clips = sample[0]
			gt_sal = sample[1]
			task_dict = sample[2]
			route_map = sample[3]
			vid_ids, frame_ids, sample_idx = [x.tolist() for x in sample[4]]

			if self.use_task:
				task = {}
				for k, v in task_dict.items():
					task[k] = v.to(self.device)

			img_clips = img_clips.to(self.device)
			img_clips = img_clips.permute((0,2,1,3,4))

			if self.use_map:
				route_map = route_map.to(self.device)
					
			with torch.no_grad():
				if self.use_task:
					pred_sal = self.model(img_clips, task)
				elif self.use_map:
					pred_sal = self.model(img_clips, route_map)
				else:
					pred_sal = self.model(img_clips, None)

			pred_sal = transforms.Resize(gt_sal.shape[-2:])(pred_sal)
			
			assert pred_sal.shape == gt_sal.shape

			gt_sal = gt_sal.to(self.device)

			pred_sal = blur_func(pred_sal)
			
			if self.test_params['evaluate']:
				self.eval_batch(frame_ids, pred_sal, gt_sal, eval_dict[vid_ids[0]])

			if self.test_params['save_images']:
				pred_sal = pred_sal.cpu().squeeze(0)
				self.save_batch(vid_ids, frame_ids, pred_sal)
	
		if self.test_params['evaluate']:
			self.save_eval_dict(eval_dict)


	def eval_batch(self, frame_idx, pred_sal, gt_sal, eval_dict):
		cc_loss = cc(pred_sal, gt_sal).cpu().numpy()
		sim_loss = sim(pred_sal, gt_sal).cpu().numpy()
		nss_loss = nss(pred_sal, gt_sal).cpu().numpy()
		kldiv_loss = kldiv(pred_sal, gt_sal).cpu().numpy()

		for i, frame_id in enumerate(frame_idx):
			eval_dict[frame_id]['CC'] = cc_loss[i]
			eval_dict[frame_id]['SIM'] = sim_loss[i]
			eval_dict[frame_id]['NSS'] = nss_loss[i]
			eval_dict[frame_id]['KLdiv'] = kldiv_loss[i]
			#eval_dict[frame_id]['sAUC'] = shuf_auc[i].cpu().numpy()


	def save_batch(self, vid_ids, frame_ids, pred_sal):

		if self.test_params['batch_size'] > 1:
			if len(pred_sal.shape) == 3:
				for idx in range(pred_sal.shape[0]):
					if self.dataset_name == 'DReyeVE':
						save_dir = join(self.results_dir, f'{vid_ids[idx]:02d}')
						save_file = join(save_dir, f'{frame_ids[idx]:06d}.png')
					else:
						save_dir = join(self.results_dir, f'{vid_ids[idx]}')
						save_file = join(save_dir, f'{frame_ids[idx]:05d}.png')

					os.makedirs(save_dir, exist_ok=True)
					
					if not os.path.exists(save_file):
						img_save(pred_sal[idx, :], save_file, normalize=True)                    
			else:
				if self.dataset_name == 'DReyeVE':
					save_dir = join(self.results_dir, f'{vid_ids[0]:02d}')
					save_file = join(save_dir, f'{frame_ids[0]:06d}.png')
				else:
					save_dir = join(self.results_dir, f'{vid_ids[0]}')
					save_file = join(save_dir, f'{frame_ids[0]:05d}.png')

				os.makedirs(save_dir, exist_ok=True)

				if not os.path.exists(save_file):
					img_save(pred_sal, save_file, normalize=True)

	def write_results(self, eval_dict):
		vid_ids = sorted(eval_dict.keys())
		eval_df = eval_dict[vid_ids[0]]

		for vid_id in tqdm(vid_ids[1:]):
			eval_df.extend(eval_dict[vid_id])        

		eval_df = pd.DataFrame.from_dict(eval_df)

		metrics = ['KLdiv', 'CC', 'SIM', 'NSS']
		eval_df = eval_df[['vid_id', 'frame_id']+metrics].dropna(axis=1, how='all') # drop columns that contain all NaNs

		# load valid frames for evaluation
		# i.e. no empty ground truth and no u-turns (in new annotations)
		evaluation_frames_df = self.datautils.get_evaluation_frames(gt='new')

		eval_df = eval_df.merge(evaluation_frames_df, how='inner', on=['vid_id', 'frame_id']).sort_values(by=['vid_id', 'frame_id'])

		df_per_video = eval_df.groupby(by=['vid_id'], dropna = True).mean()
		print('##### METRICS PER VIDEO #####')

		df_per_video.pop('frame_id')
		print(df_per_video)

		print('##### METRICS PER DATASET #####')
		df_per_dataset = df_per_video.mean()
		print(df_per_dataset)

		print('##### METRICS PER ATTRIBUTE #####')
		
		vid_attr = self.datautils.get_video_attributes()
		attr_eval_df = pd.merge(eval_df, vid_attr, how='left', left_on=['vid_id'], right_on=['vid_id'])

		df_per_attribute = {}
		for attr in ['time_of_day', 'weather', 'location']:			
			df_per_attribute[attr] = attr_eval_df[[attr] + metrics].groupby(by=[attr], dropna=True).mean()
			print(df_per_attribute[attr])

		print('##### METRICS PER ACTION #####')
		action_df = self.datautils.get_action_frames()
		action_df = pd.merge(eval_df, action_df, how='left', left_on=['vid_id', 'frame_id'], right_on=['vid_id', 'frame_id'])
		action_df = action_df[['action']+metrics].groupby(by=['action'], dropna=True).mean()
		print(action_df)

		print('##### METRICS PER CONTEXT #####')
		context_df, inters_df = self.datautils.get_intersection_frames()
		context_df = pd.merge(eval_df, inters_df, how='left', left_on=['vid_id', 'frame_id'], right_on=['vid_id', 'frame_id'])
		context_df = context_df[['intersection', 'priority']+metrics].groupby(by=['intersection', 'priority'], dropna=True).mean()
		print(context_df)

		save_path = os.path.join(self.config_dir, 'results_'+self.configs['best_weights'].replace('.pt', '.xlsx'))
		
		print(f'Saving results to {save_path}... ', end='', flush=True)
		with pd.ExcelWriter(save_path, engine='xlsxwriter') as writer:
			eval_df.to_excel(writer, sheet_name='by_frame')
			df_per_video.to_excel(writer, sheet_name='by_video')
			df_per_dataset.to_excel(writer, sheet_name='by_dataset')
			for attr, df in df_per_attribute.items():
				df.to_excel(writer, sheet_name=attr)
			action_df.to_excel(writer, sheet_name='action')
			context_df.to_excel(writer, sheet_name='context')
			print('done')

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--config_dir', required=True, type=str, help='Path to train_run')
	parser.add_argument('--evaluate', action='store_true')
	parser.add_argument('--save_images', action='store_true')
	args = parser.parse_args()

	print(args)
	print(f'-> Loading saved model from {args.config_dir}')

	test = Test()
	test.test_saved(**vars(args))
