import os
import sys
import cv2
import subprocess
import skimage.transform
import time
import argparse
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm
from dreyeve_data_utils import DReyeVEUtils
dataset_path = os.environ['DREYEVE_PATH']
scale_down_x = 4

class VisDReyeVE():
	def __init__(	self,
					video_id=1,
					etg_frame_range=None,
					gar_frame_range=None,
					show_etg_video=True,
					show_etg_gaze=True,
					show_gar_video=True,
					show_gar_gaze=False,
					show_maad_gaze=False,
					show_gt=True,
					show_old_gt=False,
					show_pred=[],
					pred_labels=[],
					show_vehicle_info=False,
					save_video_path=None,
					keep_frames=False,
					img_ext='pdf'):

		self._vid_id = video_id
		self._etg_frame_range = etg_frame_range
		self._gar_frame_range = gar_frame_range

		self._show_etg_video = show_etg_video
		self._show_etg_gaze	= show_etg_gaze
		self._show_gar_video = show_gar_video
		self._show_gar_gaze = show_gar_gaze
		self._show_maad_gaze = show_maad_gaze
		self._show_gt = show_gt
		self._show_old_gt = show_old_gt
		self._show_pred = show_pred
		self._pred_labels = pred_labels
		self._show_vehicle_info = show_vehicle_info
		self._save_video_path = save_video_path
		self._keep_frames = keep_frames

		self._img_ext = img_ext

		self._dreyeve_utils = DReyeVEUtils()


	def show_gar_frame(self, vid_id, gar_frame_idx, gar_gaze, ax):
		if gar_frame_idx is not None:
			gar_frame = mpimg.imread(os.path.join(dataset_path, f'{self._vid_id:02d}', 'frames', f'{gar_frame_idx:06d}.jpg'))
			gar_frame = skimage.transform.resize(gar_frame, (int(gar_frame.shape[0]/4), int(gar_frame.shape[1]/4)))
			ax.clear()
			ax.imshow(gar_frame)
			#print('gar resized', gar_frame.shape)
			if len(gar_gaze):
				gar_gaze = np.array(gar_gaze)
				ax.plot(gar_gaze[:, 0]//4, gar_gaze[:, 1]//4, 'r+', linewidth=10, markersize=12)
			ax.set_title(f'GAR {vid_id}:{gar_frame_idx}')
			ax.set_axis_off()

	def show_etg_frame(self, vid_id, etg_frame_idx, etg_gaze, ax):
		scale_down = 2.67
		if etg_frame_idx is not None:
			etg_frame = mpimg.imread(os.path.join(dataset_path, f'{self._vid_id:02d}', 'frames_etg', f'{etg_frame_idx:06d}.jpg'))
			#print('orig etg', etg_frame.shape)
			etg_frame = skimage.transform.resize(etg_frame, (int(etg_frame.shape[0]/scale_down), int(etg_frame.shape[1]/scale_down)))
			etg_frame = np.pad(etg_frame, ((1, 0), (121, 0), (0, 0)), mode='constant', constant_values=1) 
			#print('resized_etg', etg_frame.shape)
			ax.clear()
			ax.imshow(etg_frame)
			if len(etg_gaze):
				etg_gaze = np.array(etg_gaze)
				ax.plot(etg_gaze[:, 0]/scale_down+121, etg_gaze[:, 1]/scale_down, 'r+', linewidth=10, markersize=12)
			ax.set_title(f'           ETG {vid_id}:{etg_frame_idx}')
			ax.set_axis_off()

	def show_gt_frame(self, vid_id, frame_idx, ax, old=False):
		if old:
			gt_frame = mpimg.imread(os.path.join(dataset_path, f'{self._vid_id:02d}', 'saliency_fix', f'{frame_idx:06d}.png'))
			gt_frame = skimage.transform.resize(gt_frame, (int(gt_frame.shape[0]/4), int(gt_frame.shape[1]/4)))
			title = f'GT old {vid_id}:{frame_idx}'
		else:
			gt_frame = mpimg.imread(os.path.join(dataset_path, f'{self._vid_id:02d}', 'salmaps', f'{frame_idx:06d}.png'))
			gt_frame = skimage.transform.resize(gt_frame, (int(gt_frame.shape[0]/4), int(gt_frame.shape[1]/4)))
			title = f'GT {vid_id}:{frame_idx}'
		ax.clear()
		ax.imshow(gt_frame)
		ax.set_title(title)
		ax.set_axis_off()

	def show_pred_frame(self, dir_path, vid_id, frame_idx, ax, label):
		pred_png = os.path.join(dir_path, f'{self._vid_id:02d}', f'{frame_idx:06d}.png')
		pred_jpg = os.path.join(dir_path, f'{self._vid_id:02d}', f'{frame_idx:06d}.jpg')
		if os.path.exists(pred_png):
			pred_frame = mpimg.imread(pred_png)
		elif os.path.exists(pred_jpg):
			pred_frame = mpimg.imread(pred_jpg)
		else:
			return

		pred_frame = skimage.transform.resize(pred_frame, (int(pred_frame.shape[0]/4), int(pred_frame.shape[1]/4)))
		title = dir_path if label is None else label

		ax.clear()
		ax.imshow(pred_frame)
		ax.set_title(title)
		ax.set_axis_off()

	def vis_data(self):

		if self._save_video_path is not None:
			os.makedirs(self._save_video_path, exist_ok=True)

		# calculate number of plots to create for visualization
		num_plots = sum([self._show_etg_video, self._show_gar_video, self._show_old_gt, self._show_gt]) + len(self._show_pred)

		plt.ion()
		figure, axs = plt.subplots(nrows=1, ncols=num_plots, figsize=(num_plots*4, 3))
		#plt.subplots_adjust(wspace=0.05, hspace=0.05)
		plt.subplots_adjust(wspace=0.01, hspace=0.01)
		#plt.tight_layout()

		if self._etg_frame_range is not None:
			frame_range_df = self._dreyeve_utils.get_range(self._vid_id, self._etg_frame_range, etg=True)
		else:
			frame_range_df = self._dreyeve_utils.get_range(self._vid_id, self._gar_frame_range, etg=False)

		frame_range_df = frame_range_df.groupby(['frame_etg', 'frame_gar']).agg(list).reset_index()

		for index, row in tqdm(frame_range_df.iterrows()):
			
			ax_idx = 0
			if self._show_etg_video:
				etg_gaze = []
				if self._show_etg_gaze:
					for x, y, e, l in zip(row['X'], row['Y'], row['event_type'], row['loc']):
						if x == 'NaN' or y == 'NaN':
							continue
						if e == 'Fixation' and l in ['Scene', 'Out-of-frame']:
							etg_gaze.append([min(960, max(0, int(float(x)))), min(720, max(0, int(float(y))))])

				self.show_etg_frame(self._vid_id, row['frame_etg'], etg_gaze, axs[ax_idx])
				ax_idx += 1
			
			if self._show_gar_video:
				gar_gaze = []
				if self._show_gar_gaze:
					for x, y, e, l in zip(row['X_gar'], row['Y_gar'], row['event_type'], row['loc']):
						if x == 'NaN' or y == 'NaN':
							continue
						if e == 'Fixation' and l in ['Scene', 'Out-of-frame']:
							gar_gaze.append([int(float(x)), int(float(y))])

				self.show_gar_frame(self._vid_id, row['frame_gar'], gar_gaze, axs[ax_idx])
				ax_idx += 1
				if self._show_vehicle_info:
					pass

			if self._show_gt:
				self.show_gt_frame(self._vid_id, row['frame_gar'], axs[ax_idx], old=False)
				ax_idx += 1

			if self._show_old_gt:
				self.show_gt_frame(self._vid_id, row['frame_gar'], axs[ax_idx], old=True)
				ax_idx += 1

			for idx, img_dir in enumerate(self._show_pred):
				label = None
				if len(self._pred_labels):
					label = self._pred_labels[idx]
				self.show_pred_frame(img_dir, self._vid_id, row['frame_gar'], axs[ax_idx], label)
				ax_idx +=1

			
			figure.canvas.draw()
			figure.canvas.flush_events()

			if self._save_video_path is not None:
				
				plt.savefig(f'{self._save_video_path}/{index:04d}.{self._img_ext}')

			#time.sleep(0.1)
			plt.pause(0.01)  

		plt.close()



def get_config(config_path):
	with open(config_path, 'r') as fid:
		config = yaml.safe_load(fid)
	return config


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Visualize DReyeVE data')
	parser.add_argument('--config', help='yaml config file', type=str)
	parser.add_argument('--video_id', help='Video id', type=int)
	parser.add_argument('--etg_frame_range', help='Starting and ending frame of the driver video segment', nargs=2, type=int, default=None)
	parser.add_argument('--gar_frame_range', help='Starting and ending frame of the vehicle video segment', nargs=2, type=int, default=None)
	parser.add_argument('--show_etg_video', help='Show driver video', action='store_true')
	parser.add_argument('--show_gar_video', help='Show scene video', action='store_true')
	parser.add_argument('--show_etg_gaze', help='Plot gaze on the drivers video', action='store_true')
	parser.add_argument('--show_gar_gaze', help='Plot fixations on the scene video', action='store_true')
	parser.add_argument('--show_gt', help='Show new ground truth', action='store_true')
	parser.add_argument('--show_old_gt', help='Show old ground truth', action='store_true')
	parser.add_argument('--show_pred', help='Path(s) to directories with predictions', nargs='+', default=[])
	parser.add_argument('--pred_labels', help='Labels for predictions (must be the same lengths as list in show_pred)', nargs='+', default=[])
	parser.add_argument('--show_vehicle_info', help='Show speed and heading', action='store_true')
	parser.add_argument('--save_video_path', help='Path to save mp4 video', type=str, default=None)
	parser.add_argument('--keep_frames', help='Path to save frames (as .png)', action='store_false')
	parser.add_argument('--img_ext', help='Image extension, e.g. png, jpg, pdf', default='pdf')

	args = vars(parser.parse_args())
	print(args)
	if args.get('config', None) is not None:
		args = get_config(args.get('config'))
		

	if args['video_id'] is None:
		print('Error: Provide video id!')
		sys.exit(-1)

	assert(len(args['show_pred']) == len(args['pred_labels']))

	# checks if neither or both etg_frame_range or gar_frame_range were provided
	if (args['etg_frame_range'] is None) ^ (args['gar_frame_range'] is None):
		print(args)
		vis_dreyeve = VisDReyeVE(**args)
		vis_dreyeve.vis_data() 
	else:
		print('Error: Provide either etg_frame_range or gar_frame_range, not both!')
		sys.exit(-1)