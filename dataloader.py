import os
from os.path import join
import csv
import cv2, copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage.filters import gaussian_filter
import sys
import bz2
import json
from tqdm import tqdm
import pandas as pd
import pickle as pkl
import random
import time
from geopy.distance import geodesic as GD
from scipy.stats import entropy
import math

def KLdiv(gt, pred):
	"""
	This finds the KL-divergence between two different saliency maps when
	viewed as distributions: it is a non-symmetric measure of the information
	lost when saliencyMap is used to estimate fixationMap.

		Computer KL-divergence.
		:param pred : predicted saliency map
		:param gt : ground truth saliency map.
		:return score: float : score

	"""
	eps = np.finfo(np.float32).eps
	# the entropy function will normalize maps before computing Kld
	score = entropy(gt.flatten()+eps, pred.flatten()+eps)
	return score

def CC(gt, pred):
	"""
	This finds the linear correlation coefficient between two different
	saliency maps (also called Pearson's linear coefficient).
	score=1 or -1 means the maps are correlated
	score=0 means the maps are completely uncorrelated

	saliencyMap1 and saliencyMap2 are 2 real-valued matrices

		Computer CC score .
		:param pred : first saliency map
		:param gt : second  saliency map.
		:return score: float : score

	"""
	if not isinstance(pred, np.ndarray):
		pred = np.array(pred, dtype=np.float32)
	elif pred.dtype != np.float32:
		pred = pred.astype(np.float32)

	if not isinstance(gt, np.ndarray):
		gt = np.array(gt, dtype=np.float32)
	elif gt.dtype != np.float32:
		gt = gt.astype(np.float32)

	if pred.size != gt.size:
		pred = resize(pred, gt.shape)


	if not pred.std() or not gt.std():
		return 0

	def normalize_map_std(s_map):
		s_map -= s_map.mean()
		std = s_map.std()

		if std:
			s_map /= std

		return s_map, std == 0

	pred, sm_std_is_zero = normalize_map_std(pred)
	gt, gt_std_is_zero = normalize_map_std(gt)

	if sm_std_is_zero and not gt_std_is_zero:
		score = 0
	else:
		score = np.corrcoef(pred.flatten(),gt.flatten())[0][1]
	
	return score


random.seed(42)

class DReyeVEDataset(Dataset):

	def __init__(self,
				 use_images=True,
				 obs_length=16,
				 img_size=(224, 224),
				 map_size=(128, 128),
				 old_gt=False,
				 mode='train',
				 task_attributes=None,
				 map_params=None,
				 quick_eval=False,
				 weight_type='KLdiv'):
		'''
			use_images: use images of the scene
			obs_length: length of the input clip (default=16)
			img_size: size of image input
			old_gt: use old or new ground truth
			mode: current mode (test, train, or val)
			task_attributes: a dict of task attributes to return
			map_params: parameters for map extraction
			vid_id: video id (only for test)
			quick_eval: evaluate on 10% of the data
			weight_type: use KLdiv or CC to weight samples
		'''


		self.veh_df = {}
		self.task_context = {}
		self.maps = {}
		self.vid_attrs = {}
		self.sample_list = None
		self.word2idx = self.get_word2idx()

		self.quick_eval = quick_eval

		self.obs_img_stack = []
		self.cur_vid_id = -1

		self.task_attributes = task_attributes
		self.map_params = map_params

		self.weight_type = weight_type

		self.obs_length = obs_length
		self.gt_dir = 'saliency_fix' if old_gt else 'salmaps'
		self.mode = mode
		

		self.use_images = use_images
		self.img_size = img_size
		self.map_size = map_size


		self.img_transform = transforms.Compose([
			transforms.Resize(img_size),
			transforms.ToTensor(),
			transforms.Normalize(
				[0.485, 0.456, 0.406],
				[0.229, 0.224, 0.225]
			)
		])
		self.map_transform = transforms.Compose([
			transforms.Resize(map_size),
			transforms.ToTensor()
		])
		
		#self.list_num_frame = self.veh_df[['vid_id', 'frame']].values[::self.obs_length]

	def setup(self, test_vid_id=-1):

		self.data_path = os.environ['DREYEVE_PATH']
		self.annot_path = os.path.join(os.environ['EXTRA_ANNOT_PATH'], 'DReyeVE')
		self.dataset = 'dreyeve'

		if self.mode == "train":
			self.video_range = range(1, 35)
			self.step = self.obs_length//2

		elif self.mode=="val":
			self.video_range = range(35, 38)
			self.step = self.obs_length//2
		else:
			self.video_range = range(38, 75)
			self.step = 1

		self.frame_num = {}
		for vid_id in self.video_range:
			self.frame_num[vid_id] = 7500

		if self.mode == 'test':
			self.load_test_data(test_vid_id)
		else:
			self.load_data()


	def get_word2idx(self):
		# create a dictionary of words for actions and context
		action_context_list = ['accelerate', 'decelerate', 'stopped', 'maintain',
							   'drive straight', 'lane change left', 'lane change right', 'turn left', 'turn right',
							   'enter roundabout', 'exit roundabout', 'signalized', 'unsignalized', 'merge',
							   'right-of-way', 'yield',
							   'sunny', 'cloudy', 'rainy',
							   'downtown', 'highway', 'countryside',
							   'night', 'morning', 'evening']
		word2idx = {word: idx+1 for idx, word in enumerate(action_context_list)}
		#idx2word = {idx: word for idx, word in enumerate(action_context_list)}
		return word2idx


	def get_frame_weights(self):
		# compute KLdiv between each frame and mean of the video
		print('-> Loading sample weights...')
		weights_file = 'cache/train_frame_weights.pkl'
		if os.path.exists(weights_file):
			with open(weights_file, 'rb') as fid:
				frame_weights_dict = pkl.load(fid)
		else:
			frame_weights_dict = {}
			for vid_id in self.video_range:
				vid_dir = os.path.join(self.data_path, f'{vid_id:02d}', 'salmaps')
				mean_frame_path = os.path.join(self.annot_path, 'mean_frames', f'{self.dataset}_mean_gt_{vid_id}.png')
				mean_frame = cv2.imread(mean_frame_path, cv2.IMREAD_GRAYSCALE)
				frame_weights_dict[vid_id] = {}
				num_frames = self.frame_num[vid_id]
				for frame_id in tqdm(range(1, num_frames), desc=f'vid {vid_id}'):
					img_path = os.path.join(vid_dir, f'{frame_id:06d}.png')
					frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
					kldiv_score = KLdiv(mean_frame.copy(), frame.copy())
					cc_score = CC(mean_frame.copy(), frame.copy()) 
					#kldiv_score = np.random.randint(0.1, 10)
					frame_weights_dict[vid_id][frame_id] = (kldiv_score, cc_score)
			with open(weights_file, 'wb') as fid:
				pkl.dump(frame_weights_dict, fid)
		return frame_weights_dict

	def get_sample_weights(self):
		if self.weight_type == 'KLdiv':
			w_idx = 0
		else:
			w_idx = 1
		frame_weights_dict = self.get_frame_weights()
		sample_weights = [0]*len(self.sample_list)
		for idx, sample in enumerate(self.sample_list):
			vid_id, frame_idx, task_context_dict, route_info = sample
			for frame_id in range(frame_idx, frame_idx+self.obs_length):
				sample_weights[idx] += frame_weights_dict[vid_id][frame_id][w_idx]
			sample_weights[idx] /= self.obs_length
			if self.weight_type == 'CC':
				sample_weights[idx] =  1 - max(0.01, sample_weights[idx])
		return sample_weights

	def load_vehicle_data(self, video_range, verbose=True):

		for vid_id in tqdm(video_range, disable=(not verbose)):
			veh_data_path = f'{self.annot_path}/vehicle_data/{vid_id:02d}.xlsx'
			if not os.path.exists(veh_data_path):
				print(vid_id)
			veh_df = pd.read_excel(veh_data_path)
			veh_df.reset_index().set_index('frame')

			# label lat and long actions
			veh_df['lat action'].fillna('drive straight', inplace=True)
			
			# https://journals.sagepub.com/doi/pdf/10.3141/2663-17
			# threshold for deceleration event in data is set to -0.4
			bins = [-10000, -0.4, 0.4, 10000]
			labels = ['decelerate', 'maintain', 'accelerate']
			veh_df['lon action'] = pd.cut(veh_df['acc'], bins=bins, labels=labels).astype(str)
			veh_df.loc[(veh_df['acc'] == 0) & (veh_df['speed'] < 1), 'lon action'] = 'stopped'	
			self.veh_df[vid_id] = veh_df
			self.task_context[vid_id] = veh_df[['frame', 'lat', 'lon', 'lat_m', 'lon_m', 'context', 'lat action']].dropna()
	

	def get_gt_path(self, vid_id, frame_idx):
		return os.path.join(self.data_path, f'{vid_id:02d}', self.gt_dir, f'{frame_idx:06d}.png')

	def load_samples(self, vid_id, step):
		# check if img is nonzero
		def is_nonzero(img_path):
			nonzero = False
			if os.path.exists(img_path):
				img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
				nonzero = sum(sum(img)) != 0
				#nonzero = int(os.path.getsize(img_path) > 10709) # size of the empty image
			return nonzero

		# check if none of the actions contain U-turns
		def no_uturn(actions):
			return sum(['U-turn' in x for x in actions]) == 0

		def no_reverse(actions):
			return sum(['reverse' in x for x in actions]) == 0

		samples = []
		sample_idx_list = self.veh_df[vid_id]['frame'].values[1:-self.obs_length-1:step]
		#print(vid_id, self.veh_df[vid_id]['lat action'].unique())
		#if self.mode in ['train', 'val']:
		for frame_idx in sample_idx_list:

			gt_path = self.get_gt_path(vid_id, frame_idx+self.obs_length-1)
			sample = self.veh_df[vid_id][frame_idx-1:frame_idx-1+self.obs_length]
			#if self.mode in ['train', 'val']:
			lat_actions = sample['lat action'].values
			if is_nonzero(gt_path) and no_uturn(lat_actions) and no_reverse(lat_actions):
					#sample = self.veh_df[vid_id][start_idx-1:start_idx-1+self.obs_length]
					task_context_dict = self.get_task_context(vid_id, frame_idx, sample)
					route_info = self.get_route(vid_id, frame_idx, sample)
					#obs_img_stack = self.get_images(vid_id, frame_idx)
					samples.append((vid_id, frame_idx, task_context_dict, route_info))

		return sample_idx_list, samples

	def load_video_attributes(self, video_range):

		path_to_file = os.path.join(self.data_path, 'dr(eye)ve_design.txt')
		with open(path_to_file, 'r') as txt_file:
			lines = txt_file.readlines()
		for line in lines:
			vid_id, time_of_day, weather, location, driver_id, data_type = [x.strip().lower() for x in line.split('\t')]
			vid_id = int(vid_id)
			if vid_id in video_range:
				self.vid_attrs[vid_id] = {}
				self.vid_attrs[vid_id]['time_of_day'] = time_of_day
				self.vid_attrs[vid_id]['weather'] = weather
				self.vid_attrs[vid_id]['location'] = location


	def load_test_data(self, vid_id, cached=True):
		'''
			Load test data for each video separately for efficiency
		'''
		print(f'Loading {self.mode} data')
		print('-> Loading video attributes... ', end='', flush=True)
		self.load_video_attributes(self.video_range)
		print('done')
		
		#print('-> Loading vehicle data')
		self.load_vehicle_data(range(vid_id, vid_id+1), verbose=False)

		self.load_maps(range(vid_id, vid_id+1))

		print('-> Loading samples...', end='', flush=True)
		#for vid_id in tqdm(video_range):
		cache_filename = f'cache/{self.mode}_{self.dataset}_{vid_id:02d}_df.pkl'
		if cached and os.path.exists(cache_filename):
			with open(cache_filename, 'rb') as fid:
				samples = pkl.load(fid)
		else:
			sample_idx_list, samples = self.load_samples(vid_id, self.step)
			with open(cache_filename, 'wb') as fid:
				pkl.dump(samples, fid)
		
		if self.quick_eval:
			samples = samples[::10]
		self.sample_list = samples
		print(f'{len(self.sample_list)} samples loaded')
		#print(f'-> Loaded {len(self.sample_list)} samples')

	def load_data(self, cached=True):
	
		print(f'Loading {self.mode} data')
		self.load_video_attributes(self.video_range)
		# load data in pandas dataframe
		cached_filename = f'cache/{self.mode}_{self.dataset}_df.pkl'
		
		print('-> Loading vehicle data...')
		self.load_vehicle_data(self.video_range)
		
		self.load_maps(self.video_range)

		if cached and os.path.exists(cached_filename):
			print('-> Loading samples...', end='', flush=True)
			with open(cached_filename, 'rb') as fid:
				self.sample_list = pkl.load(fid)
		else:
			print('-> Loading samples...')

			self.sample_list = []
			tot_samples = 0

			for vid_id in tqdm(self.video_range):
				sample_idx_list, samples = self.load_samples(vid_id, self.step)
				self.sample_list.extend(samples)		
				tot_samples += len(sample_idx_list)

			with open(cached_filename, 'wb') as fid:
				pkl.dump(self.sample_list, fid)
				print(f'-> Saved {len(self.sample_list)} samples... ')

		print(f'-> Loaded {len(self.sample_list)} samples')

	def load_maps(self, video_range, debug=False):

		print(f'-> Loading maps...')
		map_info_path = os.path.join(self.annot_path, 'route_maps', 'map_info.json')
		with open(map_info_path, 'r') as fid:
			map_info = json.load(fid)

		#self.load_vehicle_data(video_range)

		for vid_id in tqdm(video_range):
			map_path = os.path.join(self.annot_path, 'route_maps', f'{vid_id:02}.png')
			self.maps[vid_id] = {}
			self.maps[vid_id]['map_path'] = map_path
			self.maps[vid_id]['map_img'] = Image.open(map_path).convert('L')
			self.maps[vid_id]['full_traj'] = Image.new(mode='L', size=self.maps[vid_id]['map_img'].size)
			#self.maps[vid_id]['map_img'] = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
			self.maps[vid_id]['ylim'] = map_info[str(vid_id)]['ylim']
			self.maps[vid_id]['xlim'] = map_info[str(vid_id)]['xlim']

			# plot the route on an empty image
			min_y, max_y = self.maps[vid_id]['ylim']
			min_x, max_x = self.maps[vid_id]['xlim']

			lat = self.veh_df[vid_id]['lat_m']
			lon = self.veh_df[vid_id]['lon_m']

			w, h = self.maps[vid_id]['map_img'].size
			
			y_px = h-((lat-min_y)*h)/(max_y-min_y) # vertical coords
			x_px = ((lon-min_x)*w)/(max_x-min_x) # horizontal coords


			draw = ImageDraw.Draw(self.maps[vid_id]['full_traj'])
			for idx, (x, y) in enumerate(zip(x_px, y_px)):
				#draw.point((int(x), int(y)), fill=255)
				draw.ellipse((int(x)-1, int(y)-1, int(x)+1, int(y)+1), fill=255)
			#self.maps[vid_id]['full_traj'].save(f'map_debug/{vid_id}_route.png')

			if debug:
				# convert the lat/lon tracks to pixel coords
				# and plot them on the image for debugging
				
				map_img = Image.open(map_path).convert('RGB')
				draw = ImageDraw.Draw(map_img)
				
				#map_img = cv2.imread(map_path, cv2.IMREAD_COLOR)

				w, h = self.maps[vid_id]['map_img'].size

				min_y, max_y = self.maps[vid_id]['ylim']
				min_x, max_x = self.maps[vid_id]['xlim']

				lat = self.veh_df[vid_id]['lat_m']
				lon = self.veh_df[vid_id]['lon_m']

				y_px = h-((lat-min_y)*h)/(max_y-min_y) # vertical coords
				x_px = ((lon-min_x)*w)/(max_x-min_x) # horizontal coords
				
				for idx, (y, x) in enumerate(zip(y_px, x_px)):
					if idx == 0:
						draw.ellipse((int(x)-1, int(y)-1, int(x)+1, int(y)+1), fill=(0, 255, 0))
						#cv2.circle(map_img, (int(x), int(y)), radius=5, color=(255,0,0), thickness=-1)
					else:
						draw.ellipse((int(x)-1, int(y)-1, int(x)+1, int(y)+1), fill=(255, 0, 0))
						#cv2.circle(map_img, (int(x), int(y)), radius=2, color=(0,255,0), thickness=-1)

				map_img.save(f'map_debug/{vid_id}.png')
				#cv2.imwrite(f'map_debug/{vid_id}.png', map_img)


		coords_2d = np.indices(self.map_size)
		coords = np.ravel_multi_index(coords_2d, dims=self.map_size)
		coords_std = np.std(coords)
		coords_mean = np.mean(coords)
		self.map_coords = np.expand_dims((coords - coords_mean)/coords_std, axis=0).astype(np.float32)

		dist_2d = coords_2d - np.array([(self.map_size[0]-1)/2, (self.map_size[0]-1/2)]).reshape((2, 1, 1))
		dist = np.sqrt((dist_2d**2).sum(axis=0))
		dist_std = np.std(dist)
		dist_mean = np.mean(dist)
		self.map_dist = np.expand_dims((dist-dist_mean)/dist_std, axis=0).astype(np.float32)


	def get_route(self, vid_id, start_idx, sample):

		w, h = self.maps[vid_id]['map_img'].size
		min_y, max_y = self.maps[vid_id]['ylim']
		min_x, max_x = self.maps[vid_id]['xlim']

		# convert lat/lon into pixel coordinates
		lat = np.array(sample['lat_m'].values)
		lon = np.array(sample['lon_m'].values)
		
		y_px = h-((lat-min_y)*h)/(max_y-min_y) # vertical coords 
		x_px = ((lon-min_x)*w)/(max_x-min_x) # horizontal coords

		m = int(len(y_px)/2)

		angle_deg = math.degrees(math.atan2(y_px[m]-y_px[-1],x_px[m]-x_px[-1]))

		return (x_px.astype(np.int), y_px.astype(np.int), angle_deg)

	def get_route_map(self, vid_id, start_idx, route_info, debug=False):
		# x,y are horz and vert coords, respectively
		# angle_deg is angle of the trajectory
		x, y, angle_deg = route_info 

		rad = self.map_params.get('radius', 50)

		x_max = x[-1] + rad
		x_min = x[-1] - rad
		y_max = y[-1] + rad
		y_min = y[-1] - rad

		rot_angle = (angle_deg + 270) % 360

		# crop map around the last point in the sample trajectory
		#map_crop = self.maps[vid_id]['map_img'][y_min:y_max, x_min:x_max]
		map_crop = self.maps[vid_id]['map_img'].crop((x_min, y_min, x_max, y_max))

		map_obs_traj = None

		if self.map_params.get('obs_traj', False):
			map_obs_traj = Image.new(mode='L', size=map_crop.size)
			draw = ImageDraw.Draw(map_obs_traj)
			for idx, (i, j) in enumerate(zip(x, y)):
				#draw.point((int(i-x_min), int(j-y_min)), fill=255)
				x_ = int(i-x_min)
				y_ = int(j-y_min)
				draw.ellipse((x_-1, y_-1, x_+1, y_+1), fill=255)
			#map_traj = map_traj.filter(ImageFilter.GaussianBlur(radius=5))
			map_obs_traj = map_obs_traj.rotate(rot_angle, resample=Image.BILINEAR)

		map_full_traj = None
		if self.map_params.get('full_traj', False):
			map_full_traj = self.maps[vid_id]['full_traj'].crop((x_min, y_min, x_max, y_max))
			map_full_traj = map_full_traj.rotate(rot_angle, resample=Image.BILINEAR)


		map_rot = map_crop.rotate(rot_angle, resample=Image.BILINEAR)

		if debug:
			#h, w = map_crop.shape
			#map_crop_debug = cv2.cvtColor(map_crop.copy(), cv2.COLOR_GRAY2RGB)
			
			w, h = map_crop.size
			save_debug =  Image.new('RGB', (w*3, h))
			map_crop_debug = self.maps[vid_id]['map_img'].crop((x_min, y_min, x_max, y_max))
			draw = ImageDraw.Draw(map_crop_debug)

			for idx, (i, j) in enumerate(zip(x, y)):
				x_ = int(i-x_min)
				y_ = int(j-y_min)
				if idx == len(x)-1:
					draw.ellipse((x_-1, y_-1, x_+1, y_+1), fill=(255))
					#draw.point((int(i-x_min), int(j-y_min)))
					#cv2.circle(map_crop_debug, (int(j-y_min), int(i-x_min)), radius=5, color=(0,0,255), thickness=-1)
				else:
					#draw.point((int(i-x_min), int(j-y_min)))
					#cv2.circle(map_crop_debug, (int(j-y_min), int(i-x_min)), radius=2, color=(0,255,0), thickness=-1)
					draw.ellipse((x_-1, y_-1, x_+1, y_+1), fill=(255) )

			#map_rot_debug = cv2.warpAffine(src=map_crop_debug, M=M, dsize=(rad*2, rad*2))
			#cv2.imwrite(f'map_debug/{vid_id}_{start_idx}_{int(angle_deg)}_{int(rot_angle)}.jpg', cv2.hconcat([map_crop_debug, map_rot_debug]))
			
			map_rot_debug = map_crop_debug.rotate(rot_angle, resample=Image.BILINEAR)
			
			map_traj = Image.new('L', (w, h))
			draw = ImageDraw.Draw(map_traj)
			for idx, (i, j) in enumerate(zip(x, y)):
				x_ = int(i-x_min)
				y_ = int(j-y_min)
				draw.ellipse((x_-1, y_-1, x_+1, y_+1), fill=255)

			map_traj = map_traj.rotate(rot_angle, resample=Image.BILINEAR)

			save_debug.paste(map_crop_debug, (0, 0))
			save_debug.paste(map_rot_debug, (w, 0))
			save_debug.paste(map_traj, (w*2, 0))
			save_debug.save(f'map_debug/{vid_id}_{start_idx}_{int(angle_deg)}_{int(rot_angle)}.jpg')


		return map_rot, map_obs_traj, map_full_traj

	def get_task_context(self, vid_id, start_idx, sample):
		next_tc = self.task_context[vid_id][self.task_context[vid_id]['frame']>start_idx]

		next_action = torch.tensor([0], dtype=torch.long)
		dist_to_inters = torch.tensor(np.ones((1, self.obs_length))*10000, dtype=torch.float32)
		#dist_to_inters = torch.tensor([0], dtype=torch.float32)
		inters_type = torch.tensor([0], dtype=torch.long)
		priority = torch.tensor([0], dtype=torch.long)
		cur_speed = sample['speed'].values

		if not next_tc.empty:
			next_tc = next_tc.iloc[0]
			try:
				next_action = torch.tensor([self.word2idx[next_tc['lat action']]], dtype=torch.long)
				
				try:
					inters_type_str, priority_str, _ = [x.strip() for x in next_tc.context.split(';')]
				except ValueError:
					inters_type_str, priority_str = [x.strip() for x in next_tc.context.split(';')]

				dist_to_inters_m = GD(sample.iloc[-1][['lat', 'lon']].values, next_tc[['lat', 'lon']].values).m
				# max lead distance = (speed in km/h * 2.22) + 37.144
				# accoridng to https://www.fhwa.dot.gov/publications/research/safety/98057/toc.cfm
				
				if dist_to_inters_m < (cur_speed.mean()*2.22)+37.144:
					#dist_to_inters = torch.tensor([dist_to_inters_m], dtype=torch.float32)
					#dist_to_inters = torch.tensor([1], dtype=torch.float32)
					dist_to_inters = torch.tensor(np.ones((1, self.obs_length))*dist_to_inters_m, dtype=torch.float32)
					inters_type = torch.tensor([self.word2idx[inters_type_str]], dtype=torch.long)
					priority = torch.tensor([self.word2idx[priority_str]], dtype=torch.long)
			except KeyError as e:
				print(vid_id, start_idx, next_tc)
				raise(KeyError, e)

		try:
			tod, weather, loc = [v for k, v in self.vid_attrs[vid_id].items()]
			cur_action = torch.tensor([self.word2idx[sample['lat action'].mode()[0]]], dtype=torch.long)
			tod = torch.tensor([self.word2idx[tod]], dtype=torch.long)
			weather = torch.tensor([self.word2idx[weather]], dtype=torch.long)
			loc = torch.tensor([self.word2idx[loc]], dtype=torch.long)
		except KeyError as e:
				print(vid_id, start_idx, next_tc)
				raise(KeyError, e)
		
		cur_acc = torch.tensor(np.expand_dims(sample['acc'].values, axis=0), dtype=torch.float32)
		cur_speed = torch.tensor(np.expand_dims(cur_speed, axis=0), dtype=torch.float32)
		lat = torch.tensor(np.expand_dims(sample['lat'].values, axis=0), dtype=torch.float32)
		lon = torch.tensor(np.expand_dims(sample['lon'].values, axis=0), dtype=torch.float32)
		
		task_context_dict = {'cur_acc': cur_acc, 'cur_speed': cur_speed, 'dist_to_inters': dist_to_inters,
						'cur_action': cur_action, 'next_action': next_action, 'inters_priority': priority, 
						'tod': tod, 'weather': weather, 'loc': loc}
		return task_context_dict		


	def get_images(self, vid_id, start_idx):
		obs_img = []
		img_dir_path = os.path.join(self.data_path, f'{vid_id:02d}', 'frames_resized')
		for i in range(self.obs_length):
			load_time = time.time()
			img = self.load_image(img_dir_path, start_idx+i)
			obs_img.append(self.img_transform(img))
		return obs_img

	def load_image(self, img_dir_path, frame_idx):
		img_path = os.path.join(img_dir_path, f'{frame_idx:06d}.jpg')
		try:
			img = Image.open(img_path).convert('RGB')
		except OSError:
			print('ERROR: could not load', img_path)
			raise(OSError)
		return img

	def get_gt(self, vid_id, start_idx):
		path_annt = os.path.join(self.data_path, f'{vid_id:02d}', self.gt_dir)
		gt = np.array(Image.open(os.path.join(path_annt, f'{start_idx+self.obs_length-1:06d}.png')).convert('L'))
		gt = gt.astype('float')
		

		if np.max(gt) > 1.0:
			gt = gt / 255.0		
		return gt

	def __len__(self):
		return len(self.sample_list)

	def __getitem__(self, idx):
		# print(self.mode)

		tot_time = time.time()
		(vid_id, start_idx, task_context_dict, route_info) = self.sample_list[idx]

		obs_img = None
		if self.use_images:
			obs_img_stack = self.get_images(vid_id, start_idx)
			obs_img = torch.FloatTensor(torch.stack(obs_img_stack, dim=0))

		gt = self.get_gt(vid_id, start_idx)

		tot_time = time.time() - tot_time
		
		# filter task attributes using model parameters
		if self.task_attributes is not None:
			task_context_dict = {k: v for k, v in task_context_dict.items() if self.task_attributes[k]}

		route_map = []

		if self.map_params is not None:
			road_map, obs_traj_map, full_traj_map = self.get_route_map(vid_id, start_idx, route_info)

			road_map = self.map_transform(road_map)
			route_map.append(road_map)

			if self.map_params.get('coords', False):
				route_map.append(torch.from_numpy(self.map_coords))
			if self.map_params.get('dist', False):
				route_map.append(torch.from_numpy(self.map_dist))
			if self.map_params.get('obs_traj', False):
				route_map.append(self.map_transform(obs_traj_map))
			if self.map_params.get('full_traj', False):
				route_map.append(self.map_transform(full_traj_map))

			#print(len(route_map), [x.shape for x in route_map])
			if len(route_map) > 1:
				route_map = torch.FloatTensor(torch.cat(route_map, dim=0))
			else:
				route_map = torch.FloatTensor(route_map[0])
			#print(route_map.shape)

		return obs_img, torch.FloatTensor(gt), task_context_dict, route_map, (vid_id, start_idx+self.obs_length-1, idx)

class BDDADataset(DReyeVEDataset):
	def __init__(self,
				 obs_length,
				 img_size=(224, 224),
				 map_size=(128, 128), 
				 mode="train",
				 task_attributes=None,
				 map_params=None,
				 **kwargs):
		
		super(BDDADataset, self).__init__()

		self.veh_df = {}
		self.task_context = {}
		self.maps = {}
		self.vid_attrs = {}
		self.sample_list = None
		self.word2idx = self.get_word2idx()

		self.task_attributes = task_attributes
		self.map_params = map_params

		self.obs_length = obs_length
		self.mode = mode
		self.img_size = img_size
		self.map_size = map_size 

		self.img_transform = transforms.Compose([
			transforms.Resize(img_size),
			transforms.ToTensor(),
			transforms.Normalize(
				[0.485, 0.456, 0.406],
				[0.229, 0.224, 0.225]
			)
		])

		self.map_transform = transforms.Compose([
			transforms.Resize(map_size),
			transforms.ToTensor()
		])


	def setup(self, vid_id=-1):

		self.dataset_path = os.environ['BDDA_PATH']
		self.annot_path = os.path.join(os.environ['EXTRA_ANNOT_PATH'], 'BDD-A')
		self.video_attr_file = os.path.join(self.annot_path, 'video_labels.xlsx')
		self.dataset = 'bdda'

		with open(f'{self.annot_path}/exclude_videos.json', 'r') as fid:
			exclude_videos = json.load(fid)

		if self.mode == "train":
			video_ids = list(set(os.listdir(os.path.join(self.dataset_path,'training','camera_frames'))) 
						- set([str(x) for x in exclude_videos['training']]))

			self.data_path = os.path.join(self.dataset_path, 'training')

			self.step = self.obs_length//2
			
		elif self.mode=="val":
			video_ids = list(set(os.listdir(os.path.join(self.dataset_path,'validation','camera_frames'))) 
						- set([str(x) for x in exclude_videos['validation']]))
			self.data_path = os.path.join(self.dataset_path, 'validation')

			self.step = self.obs_length//2
		else:
			video_ids = list(set(os.listdir(os.path.join(self.dataset_path,'test','camera_frames'))) 
						- set([str(x) for x in exclude_videos['test']]))
			self.data_path = os.path.join(self.dataset_path, 'test')

			self.step = 1

		self.video_range = sorted([int(x) for x in video_ids])
		self.frame_num = {}
		for vid_id in self.video_range:
			self.frame_num[int(vid_id)] = os.listdir(self.data_path)

		self.load_data()

	def get_word2idx(self):
		# create a dictionary of words for actions and context
		action_context_list = ['accelerate', 'decelerate', 'stopped', 'maintain',
							   'drive straight', 'lane change left', 'lane change right', 'turn left', 'turn right',
							   'enter roundabout', 'exit roundabout', 'signalized', 'unsignalized', 'merge',
							   'right-of-way', 'yield',
							   'clear', 'overcast', 'rainy',
							   'urban', 'suburban', 'highway',
							   'night', 'daytime', 'morning', 'evening']
		word2idx = {word: idx+1 for idx, word in enumerate(action_context_list)}
		#idx2word = {idx: word for idx, word in enumerate(action_context_list)}
		return word2idx

	def load_video_attributes(self, video_range):

		video_attrs_df = pd.read_excel(self.video_attr_file)
		self.vid_attrs = {}
		for idx, row in video_attrs_df.iterrows():
			self.vid_attrs[int(row['vid_id'])] = row[['time_of_day', 'weather', 'location']]


	def load_vehicle_data(self, video_range, verbose=True):

		for vid_id in tqdm(video_range, disable=(not verbose)):
			veh_data_path = f'{self.annot_path}/vehicle_data/{vid_id}.xlsx'
			veh_df = pd.read_excel(veh_data_path)
			veh_df.reset_index().set_index('frame')

			# label lat and long actions
			veh_df['lat action'].fillna('drive straight', inplace=True)
			
			# https://journals.sagepub.com/doi/pdf/10.3141/2663-17
			# threshold for deceleration event in data is set to -0.4
			bins = [-10000, -0.4, 0.4, 10000]
			labels = ['decelerate', 'maintain', 'accelerate']
			veh_df['lon action'] = pd.cut(veh_df['acc'], bins=bins, labels=labels).astype(str)
			veh_df.loc[(veh_df['acc'] == 0) & (veh_df['speed'] < 1), 'lon action'] = 'stopped'	
			self.veh_df[vid_id] = veh_df
			self.task_context[vid_id] = veh_df[['frame', 'lat', 'lon', 'lat_m', 'lon_m', 'context', 'lat action']].dropna()


	def get_frame_weights(self):
		pass

	def get_sample_weights(self):
		pass

	def get_gt_path(self, vid_id, frame_idx):
		return os.path.join(self.data_path, 'gazemap_frames', str(vid_id), f'{frame_idx:05d}.png')


	def get_images(self, vid_id, start_idx):
		obs_img = []
		img_dir_path = os.path.join(self.data_path, 'camera_frames', str(vid_id))
		for i in range(self.obs_length):
			load_time = time.time()
			img = self.load_image(img_dir_path, start_idx+i)
			obs_img.append(self.img_transform(img))
		return obs_img

	def load_image(self, img_dir_path, frame_idx):
		img_path = os.path.join(img_dir_path, f'{frame_idx:05d}.jpg')
		try:
			img = Image.open(img_path).convert('RGB')
		except OSError:
			print('ERROR: could not load', img_path)
			raise(OSError)
		return img

	def get_gt(self, vid_id, start_idx):
		path_annt = os.path.join(self.data_path, 'gazemap_frames', str(vid_id))
		gt = np.array(Image.open(os.path.join(path_annt, f'{start_idx+self.obs_length-1:05d}.png')).convert('L'))
		gt = gt.astype('float')
		
		#if self.mode == 'train':
		#	gt = cv2.resize(gt, (384, 224))

		if np.max(gt) > 1.0:
			gt = gt / 255.0		
		return gt


class LBWDataset(Dataset):
	def __init__(self, obs_length, img_size=(224, 224), mode="train"):
		''' mode: train, val, save '''
		self.data_path = os.environ['LBW_PATH']
		self.annot_path = os.environ['EXTRA_ANNOT_PATH']

		self.obs_length = obs_length
		self.mode = mode
		self.img_size = img_size
		self.img_transform = transforms.Compose([
			transforms.Resize(img_size),
			transforms.ToTensor(),
			transforms.Normalize(
				[0.485, 0.456, 0.406],
				[0.229, 0.224, 0.225]
			)
		])

		obs_seq_df = pd.read_excel(os.path.join(self.annot_path, 'LBW', 'obs_sequences.xlsx'))
		obs_seq_df = obs_seq_df[obs_seq_df['data_type'] == mode]
		self.list_num_frame = []

		# due to limitations of LBW, the sequence is only 3 frames long since it is sampled at 5Hz
		# since the input size is fixed at 16 frames
		# the first 13 frames are padded with zeros
		for idx, row in obs_seq_df.iterrows():
			item = (f"Subject{row['subj_id']:02d}_{row['vid_id']}_data", row['prev_prev_frame_id'], row['prev_frame_id'], row['frame_id'])
			self.list_num_frame.append(item)

	def __len__(self):
		return len(self.list_num_frame)

	def __getitem__(self, idx):
		(vid_id, ppi, pi, i) = self.list_num_frame[idx]

		path_clip = os.path.join(self.data_path, vid_id, 'scene_ims')
		path_annt = os.path.join(self.data_path, vid_id, 'sal_ims')

		clip_img = []
		clip_gt = []

		#print(start_idx, file_name)

		clip_idx = [ppi, pi, i]

		for i in range(self.obs_length):
			if i >= self.obs_length-3:
				img_path = os.path.join(path_clip, f'{clip_idx[i-self.obs_length+3]:08d}_scene.png')
				try:
					img = Image.open(img_path).convert('RGB')
				except OSError:
					print('ERROR:', img_path, 'not found!')
					continue
				clip_img.append(self.img_transform(img))
			else:
				clip_img.append(torch.zeros(3, self.img_size[0], self.img_size[1]))

		clip_img = torch.FloatTensor(torch.stack(clip_img, dim=0))

		gt = np.array(Image.open(os.path.join(path_annt, f'{clip_idx[-1]:08d}.png')).convert('L'))
		gt = gt.astype('float')
				
		if self.mode == "train":
			gt = cv2.resize(gt, tuple(self.img_size))
				
		if np.max(gt) > 1.0:
			gt = gt / 255.0
			
		return clip_img, torch.FloatTensor(gt), (vid_id, clip_idx[-1], idx)
