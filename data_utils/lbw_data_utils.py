# Look Both Ways (LBW) data utilities
import os
import time
#import getch
import pickle as pkl
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
#pio.kaleido.scope.mathjax = None
from matplotlib_venn import venn2, venn2_circles
from matplotlib import pyplot as plt
from os.path import join, abspath, isfile, isdir, splitext
from os import makedirs, listdir
import cv2
from PIL import Image
import yaml
from tqdm import tqdm 
import fire
from scipy.interpolate import interp1d
from tabulate import tabulate
from time import sleep

extra_annot_path = os.path.join(os.environ['EXTRA_ANNOT_PATH'], 'LBW')
dataset_path = os.environ['LBW_PATH']

class LBWUtils():

	def __init__(self, cached=True):
		self._vehicle_df = None
		self._gaze_df = None
		self._vehicle_df = None
		self._excluded_videos = []
		self._test_subj_id = [22, 25, 19, 20, 15] # subject ids used for testing
		self._val_subj_id = [16, 17] # subj ids used for validation
		self._gaze_df = self.load_gaze_data(cached=cached)
		self._head_pose_df = self.load_head_pose(cached=cached)
		self.load_vehicle_data(cached=cached)
		self._camera_calib = self.load_camera_calibration()
		#self.load_vehicle_data(cached=cached)


	def print_dataset_stats(self):

		num_subjects = len(self._gaze_df['subj_id'].drop_duplicates())
		num_videos = len(self._gaze_df[['subj_id', 'vid_id']].drop_duplicates())
		num_segments = len(self._gaze_df['segm_id'].drop_duplicates())
		num_frames = len(self._gaze_df)

		vid_length = self._gaze_df.groupby(by=['subj_id', 'vid_id']).count()['frame_id']
		vid_length_mean = vid_length.mean()
		vid_length_std = vid_length.std()

		segm_length = self._gaze_df.groupby(by='segm_id').count()['frame_id']
		segm_length_mean = segm_length.mean()
		segm_length_std = segm_length.std()

		vid_df = self._gaze_df.groupby(by=['subj_id', 'vid_id'])
		duration = vid_df['frame_id'].max() - vid_df['frame_id'].min()
		missing = duration.sum()/3 - vid_df['frame_id'].count().sum()
		missing_perc = 1-vid_df['frame_id'].count().sum()/(duration.sum()/3)

		fps = 5 # videos were subsampled to 5 Hz

		table = [['# subjects', num_subjects],
				 ['# videos', num_videos],
				 ['# segments', num_segments],
				 ['# frames', num_frames],
				 ['# missing frames', f'{int(missing)}({missing_perc*100:0.2f})'],
				 ['Video length (frames)', f'{vid_length_mean:0.2f}({vid_length_std:0.2f})'],
				 ['Video length (s)', f'{vid_length_mean/fps:0.2f}({vid_length_std/fps:0.2f})'], 
				 ['Segment length (frames)', f'{segm_length_mean:0.2f}({segm_length_std:0.2f})'],
				 ['Segment length (s)', f'{segm_length_mean/fps:0.2f}({segm_length_std/fps:0.2f})']
				]
		print(tabulate(table))

	def load_camera_calibration(self):
		with open(f'{extra_annot_path}/LBW_calib.pkl', 'rb') as fid:
			camera_calib = pkl.load(fid)
		return camera_calib

	def get_video_attributes(self):
		vid_attr_file = join(extra_annot_path, 'video_labels.xlsx')
		video_attributes = pd.read_excel(vid_attr_file)
		return video_attributes

	def label_action_frames(self):

		veh_df = self._vehicle_df.copy(deep=True)
		veh_df['lat action'].fillna('drive straight', inplace=True)
		veh_df = veh_df.drop(veh_df[(veh_df['lat action'] == 'U-turn') | (veh_df['lon action'] == 'reverse')].index)

		conditions =  [	veh_df['lon action'].eq('stopped') | veh_df['lat action'].eq('stopped'), # stopped
						veh_df['lat action'].eq('drive straight') & veh_df['lon action'].eq('maintain'), # no action
						veh_df['lat action'].ne('drive straight') & veh_df['lon action'].eq('maintain'),  # lat only
						veh_df['lat action'].eq('drive straight') & veh_df['lon action'].eq('decelerate'), # deceleration
						veh_df['lat action'].eq('drive straight') & veh_df['lon action'].eq('accelerate'), # acceleration
						veh_df['lat action'].ne('drive straight') & veh_df['lon action'].ne('maintain') # accelerating after turn is not an action	
						]
		choices = ['stopped', 'maintain', 'lat only', 'dec only', 'acc only', 'lat/lon']
		veh_df['action'] = np.select(conditions, choices, default=None)

		print(len(veh_df), veh_df.groupby(by='action')['action'].count().sum())
		print(veh_df.groupby(by='action')['action'].count()/len(veh_df)*100)

		return veh_df		

	def get_action_frames(self):
		action_frames_df = self.label_action_frames()
		action_frames_df['vid_id'] = action_frames_df['video_name']
		return action_frames_df

	def count_missing_frames(self):
		veh_df = self.label_action_frames().reset_index()
		action_types = ['stopped', 'maintain', 'lat only', 'dec only', 'acc only', 'lat/lon']
		
		action_stats = []

		prev_vid = -1
		prev_subj = -1
		prev_fid = -1
		prev_action = None
		record = None

		num_rows = len(veh_df)
		for idx, row in tqdm(veh_df.iterrows(), total=num_rows, desc='counting missing frames'):
			cur_subj = row['subj_id']
			cur_vid = row['vid_id']
			cur_fid = row['frame_id']
			cur_action = row['action']
			if (cur_action != prev_action) or (cur_vid != prev_vid) or (cur_subj != prev_subj):
				if record is not None:
					record['end_frame'] = prev_fid
					action_stats.append(record)
				record = {'vid_id': cur_vid, 'action': cur_action, 'start_frame': cur_fid, 'end_frame': cur_fid, 'num_frames': 1}
			else:
				record['num_frames'] += 1

			prev_action = cur_action
			prev_vid = cur_vid
			prev_fid = cur_fid
			prev_subj = cur_subj

		# append the last record
		action_stats.append(record)

		action_stats_df = pd.DataFrame.from_dict(action_stats)
		action_stats_df['duration'] = (action_stats_df['end_frame'] - action_stats_df['start_frame'])/3+1 # estimate actual duration (frames)

		missing_frames_df = action_stats_df[['action', 'num_frames', 'duration']].groupby(by='action', as_index=False).sum()
		missing_frames_df['missing'] = 100 - missing_frames_df['num_frames']/missing_frames_df['duration']*100

		print('Missing frames per action type')
		print(missing_frames_df)

		inters_df = veh_df[['subj_id', 'vid_id', 'segm_id', 'frame_id', 'context']].dropna()

		# check how many frames are available before the intersections
		num_prior_frames = 2*5*3 # 2s at 15fps
		tot_prior_frames = 0
		for idx, row in inters_df.iterrows():
			end_fid = row['frame_id']
			for i in range(1, 11):
				if (idx-i) >= 0 and (end_fid - veh_df.iloc[idx-i, 3]) <= num_prior_frames:
					tot_prior_frames += 1

		print('Missing frames before intersections', tot_prior_frames/(30*len(inters_df))*100)



	def load_vehicle_data(self, cached=True):
		
		print('-> Loading vehicle data...')
		cached_file = 'cache/lbw_vehicle_data.pkl'

		if cached and os.path.exists(cached_file):
			with open(cached_file, 'rb') as fid:
				self._vehicle_df = pkl.load(fid)
		else:
			data_fnames = os.listdir(f'{extra_annot_path}/vehicle_data')
			
			for data_fname in tqdm(data_fnames):

				data_path = f'{extra_annot_path}/vehicle_data/{data_fname}'

				veh_df = pd.read_excel(data_path)
				if any(['Unnamed' in x for x in veh_df.columns]):
					veh_df = veh_df[veh_df.filter(regex='^(?!Unnamed)').columns]
				veh_df['lat action'].fillna('drive straight', inplace=True)
				veh_df['lon action'].fillna('maintain', inplace=True)
				
				if veh_df['subj_id'].values[0] in self._test_subj_id:
					veh_df['data_type'] = 'test'
				elif veh_df['subj_id'].values[0] in self._val_subj_id:
					veh_df['data_type'] = 'val'
				else:
					veh_df['data_type'] = 'train'

				veh_df['video_name'] = os.path.splitext(data_fname)[0]

				if self._vehicle_df is None:
					self._vehicle_df = veh_df
				else:
					self._vehicle_df = pd.concat([self._vehicle_df, veh_df], ignore_index=True)
				
			with open(cached_file, 'wb') as fid:
				pkl.dump(self._vehicle_df, fid)

	# Find overexposed frames based on the frame brightness
	def compute_avg_brightness(self, cached=True):
		print('-> Computing avg frame brightness...')
		
		cached_file = 'cache/lbw_avg_img_data.pkl'

		if cached and os.path.exists(cached_file):
			with open(cached_file, 'rb') as fid:
				img_avg_df = pkl.load(fid)
		else:
			img_avg_df = []
			video_dirs = sorted(os.listdir(dataset_path))
			for video_dir in tqdm(video_dirs):
				img_dir = os.path.join(dataset_path, video_dir, 'scene_ims')
				for img_name in sorted(os.listdir(img_dir)):
					img_path = os.path.join(dataset_path, video_dir, 'scene_ims', img_name)
					avg = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)[200:, :].mean()
					img_avg_df.append({'video_id': video_dir, 'frame_id': int(img_name.split('_')[0]), 'avg': avg})
			img_avg_df = pd.DataFrame.from_dict(img_avg_df)
			
			with open(cached_file, 'wb') as fid:
				pkl.dump(img_avg_df, fid)


		b_std = img_avg_df['avg'].std()
		b_mean = img_avg_df['avg'].mean()

		overexposed = img_avg_df[img_avg_df['avg'] > (b_mean+1.5*b_std)]
		print(f'Overexposed {len(overexposed)} ({len(overexposed)/len(img_avg_df)})')

		for idx, row in overexposed.iterrows():
			img = cv2.imread(os.path.join(dataset_path, row['video_id'], 'scene_ims', f'{row["frame_id"]:08d}_scene.png'))
			cv2.imshow('frame', img)
			cv2.waitKey(1)


	def convert_gaze_data(self):
		""" 
		Convert LBW data from files per frame to a single excel spreadsheed per video/subject
		"""
		video_dirs = sorted(os.listdir(dataset_path))

		tot_frames = 0
		missing_frames = 0
		segm_id = -1

		for video_dir in tqdm(video_dirs):
			subj_id = int(video_dir.split('_')[0][-2:])
			vid_id = int(video_dir.split('_')[1])

			save_file = os.path.join(extra_annot_path, 'gaze_data', f'{video_dir}.xlsx')

			#if os.path.exists(save_file):
			#	continue

			gaze_dir = os.path.join(dataset_path, video_dir, 'gaze_info')
			gaze_files = sorted(os.listdir(gaze_dir))

			gaze_data_df = []

			prev_frame_id = int(gaze_files[0].split('_')[0])
			first_frame = True

			for gaze_file in gaze_files:
				
				frame_id = int(gaze_file.split('_')[0])

				if first_frame:
					segm_id += 1 
					first_frame = False
				elif frame_id - prev_frame_id > 3:
					segm_id += 1
					missing_frames += (frame_id - prev_frame_id - 3)/3

					#print(frame_id, prev_frame_id, frame_id-prev_frame_id, (frame_id - prev_frame_id - 3)/3 )

				prev_frame_id = frame_id

				with open(os.path.join(gaze_dir, gaze_file), 'r') as fid:
					data = yaml.safe_load(fid)
				#print(data)
				X, Y = data['Gaze_Loc_2D']
				
				gaze_loc_3d = [float(x) for x in data['Gaze_Loc_3D'][0].split()]
				
				if 'Left_Gaze_Dir' in data:
					left_gaze_dir = [float(x) for x in data['Left_Gaze_Dir'][0].split()]
				else:
					left_gaze_dir = None

				if 'Right_Gaze_Dir' in data:
					right_gaze_dir = [float(x) for x in data['Right_Gaze_Dir'][0].split()]
				else:
					right_gaze_dir = None

				if 'Left_2D_Eye_Loc' in data:
					left_2d_eye_loc = [float(x) for x in data['Left_2D_Eye_Loc'][0].split()]
				else:
					left_2d_eye_loc = None

				if 'Right_2D_Eye_Loc' in data:
					right_2d_eye_loc = [float(x) for x in data['Right_2D_Eye_Loc'][0].split()]
				else:
					right_2d_eye_loc = None

				if 'Left_3D_Eye_Loc' in data:
					left_3d_eye_loc = [float(x) for x in data['Left_3D_Eye_Loc'][0].split()]
				else:
					left_3d_eye_loc = None

				if 'Right_3D_Eye_Loc' in data:
					right_3d_eye_loc = [float(x) for x in data['Right_3D_Eye_Loc'][0].split()]
				else:
					right_3d_eye_loc = None

				record = {'subj_id':subj_id, 'vid_id': vid_id, 'segm_id': segm_id, 'frame_id': frame_id,
						  'X': X, 'Y': Y, # gaze loc 2D
						  'Gaze_Loc_3D': gaze_loc_3d, 
						  'Left_Gaze_Dir': left_gaze_dir,
						  'Right_Gaze_Dir': right_gaze_dir, 
						  'Left_2D_Eye_Loc': left_2d_eye_loc, 
						  'Right_2D_Eye_Loc': right_2d_eye_loc, 
						  'Left_3D_Eye_Loc': left_3d_eye_loc, 
						  'Right_3D_Eye_Loc': right_3d_eye_loc} 

				gaze_data_df.append(record)

			tot_duration = gaze_data_df[-1]['frame_id'] - gaze_data_df[0]['frame_id']

			gaze_data_df = pd.DataFrame.from_dict(gaze_data_df)
			tot_frames += len(gaze_data_df)


			gaze_data_df.to_excel(save_file)								
		print(f'Missing frames: {missing_frames}/{tot_frames} ({missing_frames/tot_frames})')

	def load_gaze_data(self, cached=True, old=False):
		
		print('-> Loading gaze data...')
		cached_file = 'cache/lbw_gaze_data.pkl'

		if cached and os.path.exists(cached_file):
			with open(cached_file, 'rb') as fid:
				gaze_df = pkl.load(fid)
		else:
			gaze_df = None
			gaze_dir = os.path.join(extra_annot_path, 'gaze_data')			
			gaze_files = sorted(os.listdir(gaze_dir))

			for gaze_file_name in tqdm(gaze_files):
				temp = pd.read_excel(os.path.join(gaze_dir, gaze_file_name), index_col=0)
				
				if gaze_df is None:
					gaze_df = temp
				else:
					gaze_df = pd.concat([gaze_df, temp], ignore_index=True)

			with open(cached_file, 'wb') as fid:
				pkl.dump(gaze_df, fid)
		return gaze_df


	# from https://github.com/Kasai2020/look_both_ways/blob/main/unisal-master/gt_saliency.py
	def get_saliency_from_gaze(self, e, g, R, K_inv, depth_rect, k):
		# Get 3D points from depth map
		#X = d(x)K_inv @ x
		X,Y = np.mgrid[0:942, 0:489]
		xy = np.vstack((X.flatten(order='C'), Y.flatten(order='C'))).T
		z = np.reshape(np.ones(489*942), ((489*942), 1))
		xyz = np.hstack((xy, z))


		xyz_3D_flat = np.dot(K_inv,xyz.T).T

		xyz_3D = np.reshape(xyz_3D_flat, (942,489,3), order='C')
		xyz_3D = np.transpose(xyz_3D,(1,0,2))

		depth_rect_mult = np.reshape(depth_rect, (489,942,1))
		xyz_3D = np.transpose(xyz_3D, (2, 0 , 1))
		depth_rect_mult = np.transpose(depth_rect_mult, (2, 0 , 1))


		xyz_3D = np.multiply(depth_rect_mult, xyz_3D)
		xyz_3D = np.transpose(xyz_3D, (1, 2, 0))
		X = xyz_3D



		X = np.reshape(X, ((489*942), 3))
		X = X - e
		X_norm = np.linalg.norm(X, axis=1)

		X = np.divide(X.T, np.reshape(X_norm,(1,(489*942))))
		X = X.T
		X = (R @ X.T).T

		s_X = X

		# Get saliency value from 3D s vectors and gaze direction g
		s_g = np.exp(k * (np.transpose(g) @ np.transpose(s_X)))
		sum = np.sum(s_g)
		s_g = s_g / sum
		s_g = s_g.reshape((489,942))

		return s_g

	# The original code for saliency maps is available at https://github.com/Kasai2020/look_both_ways/blob/main/unisal-master/gt_saliency.py
	# some variables were changed
	# gaze info is retrieved from the dataframe instead of txt files for efficiency
	# the rest of the code is the same
	def save_saliency_maps(self):

		i_T_o = np.linalg.inv(self._camera_calib['transform_in']) @ self._camera_calib['transform_out']
		R = i_T_o[:3,:3]
		R = R @ self._camera_calib['r_real']
		K_inv = np.linalg.inv(self._camera_calib['K'])

		def compute_dir(gaze_dir1, gaze_dir2):
			if str(gaze_dir1) == 'nan':
				gaze_dir = np.array(eval(str(gaze_dir2)))
			elif str(gaze_dir2) == 'nan':
				gaze_dir = np.array(eval(str(gaze_dir1)))
			else:
				gaze_dir = (np.array(eval(str(gaze_dir1))) + np.array(eval(str(gaze_dir2)))) / 2

			return gaze_dir

		tot_frames = len(self._gaze_df)

		for idx, row in tqdm(self._gaze_df.iterrows(), total=tot_frames, desc='creating salmaps'):		

			subj_id = row['subj_id']
			vid_id = row['vid_id']
			frame_id = row['frame_id']

			vid_dir = f'{dataset_path}/Subject{subj_id:02d}_{vid_id}_data'

			save_path = f'{vid_dir}/sal_ims/{frame_id:08d}.png'
			if os.path.exists(save_path):
				continue
			else:

				os.makedirs(f'{vid_dir}/sal_ims/', exist_ok=True)

				scene_depth = np.load(f'{vid_dir}/scene_depth/{frame_id:08d}_depth.npy')
				scene_img = cv2.imread(f'{vid_dir}/scene_ims/{frame_id:08d}_scene.png')

				k = 20 # this is a concentration measure for von Mises probability density used here, 1/k = sigma^2
				gaze_loc_2d = np.array([row['X'], row['Y']])
				label = compute_dir(np.array(row['Left_Gaze_Dir']), np.array(row['Right_Gaze_Dir']))
				eye_loc = compute_dir(np.array(row['Left_3D_Eye_Loc']), np.array(row['Right_3D_Eye_Loc']))

				s_g = self.get_saliency_from_gaze(eye_loc, label, R, K_inv, scene_depth, k)

				s_g_vis = (s_g / np.max(s_g)) * 255
				
				s_g_vis = np.float32(s_g_vis)
				s_g_vis = cv2.cvtColor(s_g_vis,cv2.COLOR_GRAY2RGB)
				

				cv2.imwrite(save_path, s_g_vis)

				# concatenate scene image and saliency map to view them side-by-side
				# cv2.circle(scene_img, gaze_loc_2d.astype(int), 5, (0,0,255), 3)
				# cv2.circle(s_g_vis, gaze_loc_2d.astype(int), 5, (0,0,255), 3)
				# combined = np.concatenate((scene_img, s_g_vis), axis=1)
				# combined = np.uint8(combined)
				# #out.write(combined)        
				# cv2.imshow('sal_map', combined)
				# cv2.waitKey(0)


	# Find sets of consecutive frames
	def get_obs_sequences(self, obs_length=3, gt=None):
		
		cached_file = f'{extra_annot_path}/obs_sequences.xlsx'
		if os.path.exists(cached_file):
			obs_seq_df = pd.read_excel(cached_file)
		else:
			veh_df = self._vehicle_df.copy(deep=True)
			veh_df = veh_df[(veh_df['lat action'] != 'U-turn') & (veh_df['lon action'] != 'reverse')] # remove U-turns and reversals

			veh_df['prev_frame_id'] = (veh_df['frame_id']-obs_length)
			veh_df['prev_frame'] = veh_df['frame_id'].shift(periods=1, fill_value=0) == veh_df['prev_frame_id']
			veh_df['prev_prev_frame_id'] = (veh_df['frame_id']-2*obs_length)
			veh_df['prev_prev_frame'] = veh_df['frame_id'].shift(periods=2, fill_value=0) == veh_df['prev_prev_frame_id']
			obs_seq_df = veh_df[veh_df['prev_frame'] & veh_df['prev_prev_frame']][['data_type', 'video_name', 'subj_id', 'vid_id', 'segm_id', 'frame_id', 'prev_frame_id', 'prev_prev_frame_id']]
			
			obs_seq_df.to_excel(cached_file)

		return obs_seq_df

	def get_evaluation_frames(self, obs_length=3, gt=None):
		'''
			Get frames to compute evaluation metrics on 
		'''
		obs_seq_df = self.get_obs_sequences(obs_length=3)
		eval_frames_df = obs_seq_df[obs_seq_df['data_type'] == 'test']
		eval_frames_df['vid_id'] = eval_frames_df['video_name']
		eval_frames_df = eval_frames_df[['vid_id', 'frame_id']]

		return eval_frames_df

	def get_intersection_frames(self):
		
		veh_df = self._vehicle_df.copy(deep=True)
		context_df = veh_df[['video_name', 'frame_id', 'context']].dropna()

		# get context start frame before the intersection
		context_df[['intersection', 'priority']] = context_df['context'].copy().str.split(';', n=2, expand=True)
		context_df.rename(columns={'frame_id':'end_frame'}, inplace=True)

		intersection_df = []
		for index, row in context_df.iterrows():
			end_frame = int(row['end_frame'])
			start_frame = max(0, int(end_frame - 60))
			vid_id = row['video_name']
			intersection = row['intersection'].strip()
			priority = row['priority'].strip()
			for frame in range(start_frame, end_frame+1):
				intersection_df.append({'vid_id': vid_id, 'frame_id': frame, 'intersection': intersection, 'priority': priority})

		intersection_df = pd.DataFrame.from_dict(intersection_df)
		return context_df, intersection_df


	def print_driver_action_stats(self, cached=True):
		# number of actions
		# mean (std) action durations

		# print as a table (df.to_latex())
		# solution https://towardsdatascience.com/pandas-dataframe-group-by-consecutive-same-values-128913875dba
		veh_df = self._vehicle_df.copy(deep=True)
		
		veh_df.loc[veh_df['lon action'] == 'stopped', 'lat action'] = 'stopped'

		#veh_df = self.label_action_frames(veh_df)
		#vid_ids = list(veh_df['vid_id'].unique()) 

		def get_action_stats(action_type, veh_df):
			action_counts = []
			prev_vid = -1
			prev_action = None
			record = None
			action_record = None

			num_rows = len(veh_df)
			for idx, row in tqdm(veh_df.iterrows(), total=num_rows, desc=action_type):
				cur_vid = row['vid_id']
				cur_action = row[action_type]

				if (cur_action != prev_action) or (cur_vid != prev_vid):
					if record is not None:
						action_counts.append(record)
					record = {'vid_id': cur_vid, 'action': cur_action, 'start_frame': row['frame_id'], 'end_frame': row['frame_id'], 'num_frames': 1}
				else:
					record['num_frames'] += 1

				prev_action = cur_action
				prev_vid = cur_vid

			action_counts.append(record)
			
			action_counts_df = pd.DataFrame.from_dict(action_counts)
			temp = action_counts_df[['action', 'num_frames']].groupby('action')
			action_stats_df = pd.concat([temp.count(), temp.sum(), temp.sum()/len(veh_df)*100, temp.mean(), temp.std()], axis=1)
			action_stats_df.columns = ['count', 'sum', 'perc', 'mean', 'std']

			return action_counts_df, action_stats_df

		lat_action_counts_df, lat_action_stats_df = get_action_stats('lat action', veh_df)
		lon_action_counts_df, lon_action_stats_df = get_action_stats('lon action', veh_df)

		print(lon_action_stats_df.round(2).sort_values(by=['perc'], ascending=False).to_string())
		print(lat_action_stats_df.round(2).sort_values(by=['perc'], ascending=False).to_string())

		print(lon_action_stats_df[['count', 'mean', 'std', 'perc']].round(2).sort_values(by=['perc'], ascending=False).to_latex())
		print(lat_action_stats_df[['count', 'mean', 'std', 'perc']].round(2).sort_values(by=['perc'], ascending=False).to_latex())
		
		excel_filename = 'spreadsheets/lbw_task_action.xlsx'
		

		if os.path.exists(excel_filename):
			writer = pd.ExcelWriter(excel_filename, engine='openpyxl', mode='a', if_sheet_exists='replace')
			book = writer.book			
			book.create_sheet('lon_action')
			book.create_sheet('lat_action')
		else:
			writer = pd.ExcelWriter(excel_filename, engine='openpyxl')

		lon_action_stats_df.round(2).sort_values(by=['perc'], ascending=False).to_excel(writer, sheet_name='lon_action')
		lat_action_stats_df.round(2).sort_values(by=['perc'], ascending=False).to_excel(writer, sheet_name='lat_action')
		writer.close()


	def print_context_stats(self):
		# context 
		# straight road, curved road, objects present, intersections
		# intersections: signalized/unsignalized, right-of-way/no right-of-way, type of intersection (3-way, 4-way)

		veh_df = self._vehicle_df.copy(deep=True)
		tot_frames = len(veh_df)
		inters_df = self._vehicle_df[['vid_id', 'frame_id', 'context']].dropna()#.groupby(['context'], as_index=False).count()
		inters_df[['inters_type', 'priority']] = inters_df['context'].str.split(';', expand=True)
		inters_df = inters_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

		inters_stats_df = inters_df.groupby(['inters_type', 'priority']).count()

		excel_filename = 'spreadsheets/lbw_task_action.xlsx'
		
		if os.path.exists(excel_filename):
			writer = pd.ExcelWriter(excel_filename, engine='openpyxl', mode='a', if_sheet_exists='replace')
			book = writer.book
			book.create_sheet('context')
		else:
			writer = pd.ExcelWriter(excel_filename, engine='openpyxl')
		
		inters_stats_df.to_excel(writer, sheet_name='context')
		writer.close()



# python3 lbw_data_statistics.py

if __name__ == '__main__':
	ds = LBWUtils(cached=True)
	fire.Fire(ds)