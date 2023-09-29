import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from tqdm import tqdm

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
from utils.frame_utils import writeFlow

"""
Modified demo.py to compute optical flow on a set of images

"""

DEVICE = 'cuda'

def load_image(imfile, resize=False):

    if resize:
        img = np.array(Image.open(imfile).resize((960, 540))).astype(np.uint8)
    else:
        img = np.array(Image.open(imfile)).astype(np.uint8)

    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=1)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        sys.exit(-1)


def run(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        for imfile1, imfile2 in tqdm(zip(images[:-1], images[1:])):
            if args.results_path is not None:
                flofile = os.path.join(args.results_path, os.path.basename(imfile2).replace('.jpg', '.flo'))
                if os.path.exists(flofile):
                    continue
            image1 = load_image(imfile1, resize=args.resize)
            image2 = load_image(imfile2, resize=args.resize)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            
            if args.results_path is not None:
                flow_up = flow_up[0].permute(1, 2, 0).cpu().numpy()
                writeFlow(flofile, flow_up)
            else:
                viz(image1, flow_up)



# Run inside docker
# python compute_optical_flow.py --model=models/raft-things.pth --path=usr_data/DREYEVE/07/frames/ --results_path=usr_data/DREYEVE/07/flow_RAFT

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--results_path', help='path to results directory, if not provided, results will not be saved')
    parser.add_argument('--resize', action='store_true', help='downscale image x2 to simplify calculation')
    args = parser.parse_args()
    print(f'{args.results_path}')
    if args.results_path is not None:
        os.makedirs(args.results_path, exist_ok=True)
    run(args)
