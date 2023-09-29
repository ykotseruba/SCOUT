import os
import time
import sys
import json
import cv2
import argparse
import pandas as pd
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from optflow_utils import readFlow 
from tqdm import tqdm
#from scipy.ndimage import gaussian_filter
from cupyx.scipy.ndimage import gaussian_filter

# Generate motion-compensated ground truth

class ComputeGT():

	def __init__(self, seq_num, visualize=False, overwrite=False, scale_factor=2, sigma=60):

		self.dreyeve_data_root = os.environ['DREYEVE_PATH']
		self.frames_dir = os.path.join(self.dreyeve_data_root, f'{seq_num:02d}', 'frames')
		self.etg_frames_dir = os.path.join(self.dreyeve_data_root, f'{seq_num:02d}', 'frames_etg')
		self.salmap_dir = os.path.join(self.dreyeve_data_root, f'{seq_num:02d}', 'salmaps')
		self.old_salmap_dir = os.path.join(self.dreyeve_data_root, f'{seq_num:02d}', 'saliency_fix')
		#optflow_path = os.path.join(dreyeve_data_root, f'{seq_num:02d}', 'flow')

		# there was not enough space on a single hard drive
		if seq_num < 60:
			self.optflow_dir = f'/media/yulia/Storage1/DReyeVE/flow/{seq_num:02d}'
		elif seq_num == 74:
			self.optflow_dir = f'/media/yulia/Storage 5/DReyeVE/flow/{seq_num:02d}'
		else:
			self.optflow_dir = f'/media/yulia/Storage4/DReyeVE/flow/{seq_num:02d}'

		# make a directory to save new ground truth
		os.makedirs(self.salmap_dir, exist_ok=True)

		# constants for ground truth generation
		self.num_frames = 7500
		self.img_h = 1080
		self.img_w = 1920

		# if optical flow maps are resized by a factor of 2
		self.scale_factor = scale_factor

		# size of the Gaussian filter to smooth fixations
		self.sigma=sigma
		self.filter_size = 2*self.sigma*3+1
		self.T = 12 # half of temporal window for aggregating fixations

		self.gaze_data_path = f'extra_annotations/DReyeVE/gaze_data/{seq_num:02d}.txt'
		self.gaze_df = None
		self.etg2gar_df = None
		self.load_gaze_data()

		self.gt_df = [] # save fixations in the gt

		# path to save aggregated gaze
		self.new_gt_path = f'extra_annotations/DReyeVE/new_gt/{seq_num:02d}.pkl'

		self.visualize = visualize
		self.setup_vis()

		self.overwrite = overwrite

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

	def fix_within_bounds(self, fix):
		return (fix[1] >= 0) and (fix[0] >= 0) and (fix[0] < self.img_w) and (fix[1] < self.img_h)

	def setup_vis(self):
		if self.visualize:
			plt.ion()
			self.fig, self.axs = plt.subplots(1, 3, figsize=(15, 5))

	def visualize_results(self, etg_frame_id, etg_fix, frame_id):
		if self.visualize:
			if etg_frame_id > 0:
				etg_frame = cv2.imread(os.path.join(self.etg_frames_dir, f'{etg_frame_id:06d}.jpg'))[...,::-1]
			frame = cv2.imread(os.path.join(self.frames_dir, f'{frame_id:06d}.jpg'))[...,::-1]
			old_gt = cv2.imread(os.path.join(self.old_salmap_dir, f'{frame_id:06d}.png'))
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

			vis(self.axs[1], f'old gt {frame_id}', cv2.addWeighted(frame, 0.4, old_gt, 0.6, 0))
			vis(self.axs[2], f'new gt {frame_id}', cv2.addWeighted(frame, 0.4, sal_map, 0.6, 0))


			plt.draw()
			plt.pause(0.01)

	def get_displacement(self, opt_flow, fix):

		y = int(fix[0]/self.scale_factor)
		x = int(fix[1]/self.scale_factor)
		
		disp = opt_flow[x, y, :].astype(int)
		return disp

	def propagate_fixations(self, cur_f, f1, f2, step, sal_map):
		'''
		Propagate fixations across frames in range [f1, f2).
		Args:
			cur_f: index of the current frame
			f1, f2: range of frames
			step: direction in which to propagate -1 or 1
		'''
		proj_fix = []

		for k in range(f1, f2, step):

			# load optical flow
			opt_flow = readFlow(os.path.join(self.optflow_dir, f'{k:06d}.flo'))

			# load fixations in the current frame
			cur_fix = self.gaze_df[self.gaze_df['frame_gar'] == k]
			#print(cur_fix)
			if not cur_fix.empty:
				# save each fixation as a list of [x, y, location, update]
				# out-of-frame fixations are not updated and are propagated to next frames
				cur_fix = [x+[True] for x in cur_fix[['X_gar', 'Y_gar', 'loc']].values.tolist()]
				proj_fix.extend(cur_fix)

				
			# project points to the preceding frame using optical flow
			for fix in proj_fix:
				# if fixation is not out of frame or on the mirrors and can be updated
				if (fix[2] != 'Out-of-frame') and ('mirror' not in fix[2]) and fix[3]:
					disp = self.get_displacement(opt_flow, fix)
					
					# if non-zero displacement
					# propagate fixation
					if any(disp):
						fix[0] += disp[0]*step
						fix[1] += disp[1]*step
						# if fixation goes out of image bounds, it is not included in the current frame
						fix[3] = self.fix_within_bounds(fix)

		for fix in proj_fix:
			if fix[3]:
				self.gt_df.append({'frame_id': cur_f, 'X_gar': fix[0], 'Y_gar': fix[1], 'type': 'other'})
				sal_map[fix[1], fix[0]] = 1

	def add_current_fixations(self, frame_id, sal_map):
		etg_frame_id = self.etg2gar_df[self.etg2gar_df['frame_gar'] == frame_id]
		if etg_frame_id.empty:
			etg_frame_id = -1
		else:
			etg_frame_id = etg_frame_id.values.tolist()[0][0]

		cur_fix = self.gaze_df[self.gaze_df['frame_gar'] == frame_id]
		if not cur_fix.empty:
			etg_fix = cur_fix[['X', 'Y']].values.tolist()
			cur_fix = cur_fix[['X_gar', 'Y_gar', 'loc']].values.tolist()
		else:
			etg_fix = []
			cur_fix = []

		for fix in cur_fix:
			#print(fix)
			sal_map[fix[1], fix[0]] = 1
			self.gt_df.append({'frame_id': frame_id, 'X_gar': fix[0], 'Y_gar': fix[1], 'type': 'cur'})
		return etg_frame_id, etg_fix

	def generate_gt(self):
		"""Generate motion-compensated ground (requires optical flow)
		"""
		
		for frame_id in tqdm(range(1, self.num_frames)):

			save_path = os.path.join(self.salmap_dir, f'{frame_id:06d}.png')
			
			if os.path.exists(save_path) and not self.overwrite:
				if self.visualize:
					etg_fix = self.gaze_df[self.gaze_df['frame_gar'] == frame_id]
					if len(etg_fix):
						etg_fix = etg_fix[['X', 'Y']].values.tolist()
					else:
						etg_fix = []
					etg_frame_id = self.etg2gar_df[self.etg2gar_df['frame_gar'] == frame_id]
					if etg_frame_id.empty:
						etg_frame_id = -1
					else:
						etg_frame_id = etg_frame_id.values.tolist()[0][0]
					self.visualize_results(etg_frame_id, etg_fix, frame_id)
				continue

			sal_map = np.zeros((self.img_h, self.img_w))

			# add fixations from current frame
			etg_frame_id, etg_fix = self.add_current_fixations(frame_id, sal_map)

			f1 = max(1, frame_id - self.T)
			f2 = min(self.num_frames, frame_id + self.T+1)

			# propagate fixations from future frames backwards to the current one
			self.propagate_fixations(frame_id, f1, frame_id, 1, sal_map)

			# propagate fixations from previous frames towards current one
			self.propagate_fixations(frame_id, frame_id+1, f2, -1, sal_map)

			# apply Gaussian blur
			# this takes forever 
			# https://www.peterkovesi.com/matlabfns/#integral
			# https://github.com/bfraboni/FastGaussianBlur
			# https://hal.inria.fr/inria-00074778/document
			# https://www.freecodecamp.org/news/how-to-create-and-upload-your-first-python-package-to-pypi/

			#t = time.time()
			
			with cp.cuda.Device(0):
				sal_map_gpu = cp.asarray(sal_map)
				filt_sal_map_gpu = gaussian_filter(sal_map_gpu, self.sigma)
				sal_map = cp.asnumpy(filt_sal_map_gpu)


			#print(f'gauss {(time.time()-t)*1000}')

			#sal_map = cv2.GaussianBlur(sal_map, (self.filter_size, self.filter_size), self.sigma)

			plt.imsave(save_path, sal_map, cmap='Greys_r')

			self.visualize_results(etg_frame_id, etg_fix, frame_id)


		# save fixations
		self.gt_df = pd.DataFrame.from_dict(self.gt_df)
		self.gt_df.to_pickle(self.new_gt_path)

# Run from the top directory
# python3 data_labeling/create_new_ground_truth.py <seq_num>

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
	gt = ComputeGT(args.seq_num, visualize=args.visualize, overwrite=args.overwrite, scale_factor=args.scale_factor, sigma=args.sigma)
	gt.generate_gt()