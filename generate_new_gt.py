import os
import time
import sys
import json
import cv2
import argparse
import pandas as pd
import numpy as np
import cupy as cp
import pickle as pkl
import matplotlib.pyplot as plt
from tqdm import tqdm
from cupyx.scipy.ndimage import gaussian_filter

# Generate motion-compensated ground truth for DReyeVE

class GenerateGT():

	def __init__(self, seq_num, visualize=False, overwrite=False, scale_factor=2, sigma=60):

		self.dreyeve_data_root = os.environ['DREYEVE_PATH']
		self.frames_dir = os.path.join(self.dreyeve_data_root, f'{seq_num:02d}', 'frames')
		self.etg_frames_dir = os.path.join(self.dreyeve_data_root, f'{seq_num:02d}', 'frames_etg')
		self.salmap_dir = os.path.join(self.dreyeve_data_root, f'{seq_num:02d}', 'salmaps')
		self.orig_salmap_dir = os.path.join(self.dreyeve_data_root, f'{seq_num:02d}', 'saliency_fix')

		# make a directory to save new ground truth
		os.makedirs(self.salmap_dir, exist_ok=True)

		# constants for ground truth generation
		self.seq_num = seq_num
		self.num_frames = 7500
		self.img_h = 1080
		self.img_w = 1920

		# optical flow maps were resized by a factor of 2 for efficiency
		self.scale_factor = scale_factor

		# size of the Gaussian filter to smooth fixations
		self.sigma=sigma
		self.filter_size = 2*self.sigma*3+1
		self.T = 12 # half of temporal window for aggregating fixations

		self.gaze_data_path = f'extra_annotations/DReyeVE/gaze_data/{seq_num:02d}.txt'
		self.gaze_df = None
		self.etg2gar_df = None
		self.load_gaze_data()
		self.gt_df = None # dataframe for fixations

		# path to aggregated fixations
		self.new_gt_path = f'extra_annotations/DReyeVE/new_gt/{seq_num:02d}.pkl'
		self.load_gt_fixations()

		self.visualize = visualize
		self.setup_vis()

		self.overwrite = overwrite

	def load_gt_fixations(self):
		"""Load gt fixations for each frame
		"""
		with open(self.new_gt_path, 'rb') as fid:
			self.gt_df = pkl.load(fid)


	def setup_vis(self):
		if self.visualize:
			plt.ion()
			self.fig, self.axs = plt.subplots(1, 3, figsize=(15, 5))

	def visualize_results(self, etg_frame_id, etg_fix, frame_id):
		if self.visualize:
			if etg_frame_id > 0:
				etg_frame = cv2.imread(os.path.join(self.etg_frames_dir, f'{etg_frame_id:06d}.jpg'))[...,::-1]
			frame = cv2.imread(os.path.join(self.frames_dir, f'{frame_id:06d}.jpg'))[...,::-1]
			orig_gt = cv2.imread(os.path.join(self.orig_salmap_dir, f'{frame_id:06d}.png'))
			sal_map = cv2.imread(os.path.join(self.salmap_dir, f'{frame_id:06d}.png'))
			def vis(ax, title, img):
				ax.clear()
				ax.axis('off')
				ax.set_title(title)
				ax.imshow(img)
			if etg_frame_id > 0:
				vis(self.axs[0], f'etg_frame {etg_frame_id}', etg_frame)
				for fix in etg_fix:
					self.axs[0].plot(min(max(0, fix[0]), 959), min(max(0, fix[1]), 719), 'ro')

			vis(self.axs[1], f'orig gt {frame_id}', cv2.addWeighted(frame, 0.4, orig_gt, 0.6, 0))
			vis(self.axs[2], f'new gt {frame_id}', cv2.addWeighted(frame, 0.4, sal_map, 0.6, 0))


			plt.draw()
			plt.pause(0.01)

	def plot_fixations(self, cur_f, sal_map):
		
		cur_df = self.gt_df[self.gt_df['frame_id'] == cur_f]
		for idx, row in cur_df.iterrows():
			sal_map[row['Y_gar'], row['X_gar']] = 1

		return len(cur_df)

	def load_gaze_data(self):
		"""Load gaze data from a text file, remove saccades, blinks,
		and in-vehicle fixations
		"""
		gaze_df = pd.read_csv(self.gaze_data_path, sep=' ')
		self.etg2gar_df = gaze_df[['frame_etg', 'frame_gar']]

		# filter out saccades, blinks and in-vehicle data points
		gaze_df = gaze_df[(gaze_df['event_type'] == 'Fixation') & ((gaze_df['loc'] == 'Scene') | (gaze_df['loc'] == 'Out-of-frame') | (gaze_df['loc'].str.contains(r'mirror')))]
		
		# convert fixation coordinates to integers to refernece pixels in the image
		gaze_df['X_gar'] = gaze_df['X_gar'].astype(int)-1 # MATLAB indexes fixations starting at 1
		gaze_df['Y_gar'] = gaze_df['Y_gar'].astype(int)-1
		gaze_df['X'] = gaze_df['X'].astype(int)-1
		gaze_df['Y'] = gaze_df['Y'].astype(int)-1

		self.gaze_df = gaze_df[['frame_etg', 'X', 'Y', 'frame_gar',  'X_gar', 'Y_gar', 'loc']]

	def generate_gt(self):
		"""
		Generate ground truth maps
		"""

		for frame_id in tqdm(range(1, self.num_frames)):

			save_path = os.path.join(self.salmap_dir, f'{frame_id:06d}.png')

			if not os.path.exists(save_path) and not self.overwrite:
				sal_map = np.zeros((self.img_h, self.img_w))

				num_fix = self.plot_fixations(frame_id, sal_map)
				
				if num_fix > 0:
					with cp.cuda.Device(0):
						sal_map_gpu = cp.asarray(sal_map)
						filt_sal_map_gpu = gaussian_filter(sal_map_gpu, self.sigma)
						sal_map = cp.asnumpy(filt_sal_map_gpu)

				plt.imsave(save_path, sal_map, cmap='Greys_r')


			if self.visualize:
				etg_fix = self.gaze_df[self.gaze_df['frame_gar'] == frame_id]
				if len(etg_fix):
					etg_fix = etg_fix[['X', 'Y']].values.tolist()
				else:
					etg_fix = []
				etg_frame_id = self.etg2gar_df[self.etg2gar_df['frame_gar'] == frame_id]
				if not etg_frame_id.empty:
					etg_frame_id = etg_frame_id.values.tolist()[0][0]
				else:
					etg_frame_id = -1
				self.visualize_results(etg_frame_id, etg_fix, frame_id)


if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--seq_num', type=int, help='sequence number')
	parser.add_argument('--visualize', action='store_true', default=False, help='visualize the results')
	parser.add_argument('--overwrite', action='store_true', default=False, help='overwrite existing ground truth')
	parser.add_argument('--scale_factor', type=int, default=2, help='scaling factor for optical flow')
	parser.add_argument('--sigma', type=int, default=60, help='sigma for gaussian filter')
	args = parser.parse_args()

	print(args)
	print(f'Processing sequence {args.seq_num}...')
	gt = GenerateGT(args.seq_num, visualize=args.visualize, overwrite=args.overwrite, scale_factor=args.scale_factor, sigma=args.sigma)
	gt.generate_gt()