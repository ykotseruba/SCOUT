import os
import time
import pickle as pkl
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from matplotlib_venn import venn2, venn2_circles
from matplotlib import pyplot as plt
from os.path import join, abspath, isfile, isdir, splitext
from os import makedirs, listdir
import cv2
from tqdm import tqdm 
import fire
import scipy.io as sio
from tabulate import tabulate


extra_annot_path = os.path.join(os.environ['EXTRA_ANNOT_PATH'], 'DReyeVE')
dataset_path = os.environ['DREYEVE_PATH']

class DReyeVEUtils():
	def __init__(self, cached=True):
		self._vehicle_df = None
		self._excluded_videos = [6]
		self._gaze_df = self.load_gaze_data(cached=cached)
		self._old_gaze_df = self.load_gaze_data(cached=cached, old=True)
		self._empty_df = None # empty frames in old and new DReyeVE
		self._orig_action_annot_df = None # original subsequence annotations from DReyeVE
		self.load_vehicle_data(cached=cached)
		#xself.load_orig_action_annotations(cached=cached)
		self.load_empty_gt_data(cached=cached)
		self.etg_gar = self._gaze_df[['vid_id', 'frame_etg', 'frame_gar']].drop_duplicates(subset=['vid_id', 'frame_gar'])
		

	def print_dataset_stats(self):
		"""
		number of videos
		number of frames

		"""

		table = [['# videos', 74],
				 ['# frames', 74*7500]					
				]
		print(tabulate(table))


	# get data by range of etg or gar frames
	def get_range(self, vid_id, frame_range, etg=True):

		if etg:
			frame_col = 'frame_etg'
		else:
			frame_col = 'frame_gar'

		frame_range_df = self._gaze_df[(self._gaze_df['vid_id'] == vid_id) &
								(self._gaze_df[frame_col] >= frame_range[0]) & 
								(self._gaze_df[frame_col] <= frame_range[1])]
		frame_range_df = pd.merge(frame_range_df, self._vehicle_df, how='left', left_on=['vid_id', frame_col], right_on=['vid_id', 'frame_id'])

		return frame_range_df


	def compute_mean_frame(self, video_type='frame_id', old_annot=False):
		"""Compute mean image for each video in the dataset
			driver, gt, frame
		
		Args:
			video_type (str): frame (scene view), driver (driver view) or gt (saliency)
		"""
		print(f'Computing mean {video_type}...')

		if video_type == 'gt':
			if old_annot:
				mean_frame_path = join(extra_annot_path, 'mean_frames', f'dreyeve_old_mean_{video_type}.png')
				gt_dir = 'saliency_fix'
				video_mean_path_template = join(extra_annot_path, 'mean_frames', f'dreyeve_old_mean_gt')
			else:
				mean_frame_path = join(extra_annot_path, 'mean_frames', f'dreyeve_mean_{video_type}.png')
				video_mean_path_template = join(extra_annot_path, 'mean_frames', f'dreyeve_mean_gt')
				gt_dir = 'salmaps'
		else:
			mean_frame_path = join(extra_annot_path, 'mean_frames', f'dreyeve_mean_{video_type}.png')
			video_mean_path_template = join(extra_annot_path, 'mean_frames', f'dreyeve_mean_{video_type}')

		if os.path.exists(mean_frame_path):
			return cv2.imread(mean_frame_path, cv2.IMREAD_GRAYSCALE if video_type == 'gt' else cv2.IMREAD_COLOR)

		mean_frame = None

		video_dirs = [join(dataset_path, d) for d in sorted(listdir(dataset_path)) if isdir(join(dataset_path, d))]    
		for i, video_dir in enumerate(video_dirs):

			video_mean_path = f'{video_mean_path_template}_{i+1}.png'
			if os.path.exists(video_mean_path):
				video_mean_img = cv2.imread(video_mean_path)
			else:
				video_mean_img = None
				#print('\nProcessing video {}...'.format(i))
				if video_type == 'frame_id':
					img_dir = join(video_dir, 'frames')
				elif video_type == 'gt':
					img_dir = join(video_dir, gt_dir)
				elif video_type == 'driver':
					img_dir = join(video_dir, 'frames_etg')

				frames = [join(img_dir, x) for x in listdir(img_dir)]
				for frame in tqdm(frames, desc=f'vid {i+1}'):
					try:
						if video_mean_img is None:
							video_mean_img = cv2.imread(frame).astype(float)
						else:
							video_mean_img += cv2.imread(frame).astype(float)
					except:
						print(f'Error: could not load {frame}!')
						return  
				
				video_mean_img /= len(frames)

				cv2.imwrite(video_mean_path, video_mean_img.astype(int))
				print(f'Saved mean img to {video_mean_path}...')

			if mean_frame is None:
				mean_frame = video_mean_img.astype(float)
			else:
				mean_frame += video_mean_img.astype(float)

		mean_frame /= len(video_dirs)

		cv2.imwrite(mean_frame_path, mean_frame.astype(int))
		print(f'Saved mean img to {mean_frame_path}')

		return mean_frame


	def get_video_attributes(self):
		video_attributes = []

		path_to_file = join(dataset_path, 'dr(eye)ve_design.txt')
		with open(path_to_file, 'r') as txt_file:
			lines = txt_file.readlines()

		for line in lines:
			video_id, time_of_day, weather, location, driver_id, data_type = [x.strip() for x in line.split('\t')]
			video_attributes.append({'vid_id': int(video_id),
									'time_of_day': time_of_day,
									'weather': weather,
									'location': location, 
									'driver': int(driver_id[1])})

		return pd.DataFrame.from_dict(video_attributes)

	def load_orig_action_annotations(self, cached=True):
		'''
		Load original DReyeVE action annotations
		Format in subsequences
		vid_id start_frame_id end_frame_id type
		Types are:
		i - inattentive
		k - attentive and not trivial
		u - attentive but uninteresting
		e - errors
		'''
		cached_file = 'cache/dreyeve_orig_actions.pkl'
		if cached and os.path.exists(cached_file):
			with open(cached_file, 'rb') as fid:
				self._orig_annot_df = pkl.load(fid)
		else:
			self.orig_annot_df = []
			data_path = f'{dataset_path}/subsequences.txt'
			with open(data_path, 'r') as fid:
				lines = fid.readlines()
			for line in lines:
				line = line.strip().split('\t')
				vid_id, start_frame, end_frame, action_type = line
				for i in range(int(start_frame), int(end_frame)+1):
					self.orig_annot_df.append({'vid_id': int(vid_id), 'frame_id': i, 'type': action_type})
			self._orig_action_annot_df = pd.DataFrame.from_dict(self.orig_annot_df)
			with open(cached_file, 'wb') as fid:
				pkl.dump(self._orig_action_annot_df, fid)


	def get_eval_dict(self, cache_path=None):
		"""
		Load cached evaluation dict to continue or create a new one 
		"""
		eval_dict = {}
		if os.path.exists(cache_path):
			with open(cache_path, 'rb') as fid:
				eval_dict = pkl.load(fid)

			print(f'Loaded eval dict from {cache_path}')
		else:
			for vid in range(38, 75):
				eval_dict[vid] = []
				for frame_id in range(1, 7500):
					eval_dict[vid].append({'vid_id': vid,
									  'frame_id': frame_id,
									  #'pred_img_path': pred_img_path,
									  #'pred_img_mtime': pred_img_mtime,
									  #'gt_img_path': gt_img_path,
									  'SIM': None,
									  'IG': None,
									  'AUC': None,
									  'sAUC': None,
									  'KLdiv': None,
									  'KLdiv_1': None,
									  'NSS': None,
									  'CC': None})

			cache_dir = os.path.split(cache_path)[0]
			if not os.path.exists(cache_dir):
				os.makedirs(cache_dir, exist_ok=True)

			with open(cache_path, 'wb') as fid:
				pkl.dump(eval_dict, fid, protocol=pkl.HIGHEST_PROTOCOL)
			print(f'Saved eval dict to {cache_path}')
		return eval_dict


	def load_empty_gt_data(self, cached=True):
		'''
		indices of empty ground truth frames (orig and new)
		creates a dataframe with status for old and new gt
		-1 - file does not exist
		1 - salmap is nonzero
		0 - salmap is all zeros (empty)
		'''
		print('-> Identifying "empty" gt data (blank salmaps)...')
		cached_file = 'cache/dreyeve_gt_data.pkl'
		if cached and os.path.exists(cached_file):
			with open(cached_file, 'rb') as fid:
				self._empty_df = pkl.load(fid)
		else:
			def nonzero(img_path):
				nonzero = -1
				if os.path.exists(img_path):
					#img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
					#nonzero = sum(sum(img)) != 0
					nonzero = int(os.path.getsize(img_path) > 10709) # size of the empty image
				return nonzero

			self._empty_df = []
			for vid_id in tqdm(range(1, 75)):
				if vid_id in self._excluded_videos:
					continue
				old_gt_dir = os.path.join(dataset_path, f'{vid_id:02}', 'saliency_fix')	
				new_gt_dir = os.path.join(dataset_path, f'{vid_id:02}', 'salmaps')
				old_gt_frames = [os.path.join(old_gt_dir, x) for x in os.listdir(old_gt_dir) if (x.endswith('.jpg') or x.endswith('.png'))]
				new_gt_frames = [os.path.join(new_gt_dir, x) for x in os.listdir(new_gt_dir) if (x.endswith('.jpg') or x.endswith('.png'))]

				for idx, (old_gt_path, new_gt_path) in enumerate(zip(old_gt_frames, new_gt_frames)):
					self._empty_df.append({'vid_id': vid_id, 'frame_id': idx, 
										   'old_gt_status': nonzero(old_gt_path),
										   'new_gt_status': nonzero(new_gt_path)})
			self._empty_df = pd.DataFrame.from_dict(self._empty_df)
			with open(cached_file, 'wb') as fid:
				pkl.dump(self._empty_df, fid)


	def load_vehicle_data(self, cached=True):
		print('-> Loading vehicle data...')
		cached_file = 'cache/dreyeve_vehicle_data.pkl'
		if cached and os.path.exists(cached_file):
			with open(cached_file, 'rb') as fid:
				self._vehicle_df = pkl.load(fid)
		else:
			for vid_id in tqdm(range(1, 75)):
				if vid_id in self._excluded_videos:
					continue
				data_path = f'{extra_annot_path}/vehicle_data/{vid_id:02d}.xlsx'
				#print(f'Loading {data_path}...')
				veh_df = pd.read_excel(data_path)
				veh_df['vid_id'] = vid_id
				# remove unnamed columns
				if any(['Unnamed' in x for x in veh_df.columns]):
					veh_df = veh_df.loc[:, ~veh_df.columns.str.contains('^Unnamed')]
				if self._vehicle_df is None:
					self._vehicle_df = veh_df
				else:
					self._vehicle_df = pd.concat([self._vehicle_df, veh_df], ignore_index=True)
				#self._vehicle_df_list.append(pd.read_excel(data_path))
					
			self._vehicle_df = self._vehicle_df.rename(columns={'frame': 'frame_id'})
			with open(cached_file, 'wb') as fid:
				pkl.dump(self._vehicle_df, fid)

	def fix_gaze_data(self):
		print('-> Fixing gaze data...')
		for vid_id in tqdm(range(1, 75)):
			if vid_id in self._excluded_videos:
				continue
			data_path = f'{extra_annot_path}/gaze_data/{vid_id:02d}.txt'
			with open(data_path, 'r') as fid:
				lines = fid.readlines()
			# header: frame_etg frame_gar X Y X_gar Y_gar event_type code loc
			with open(data_path, 'w') as fid:
				for line in lines:		
					line = line.strip().split()
					if line[6] == '-':
						line[8] = 'NA'
					elif line[6] in ['Saccade', 'Blink']:
						line[8] = 'NA'
					elif line[6] == 'Fixation' and line[8] == 'Error':
						line[8] = 'NA'
					line = ' '.join(line)
					fid.write(line+'\n')

	def load_gaze_data(self, cached=True, old=False):
		print('-> Loading gaze data...')
		if old:
			cached_file = 'cache/dreyeve_old_gaze_data.pkl'
		else:
			cached_file = 'cache/dreyeve_gaze_data.pkl'

		if cached and os.path.exists(cached_file):
			with open(cached_file, 'rb') as fid:
				gaze_df = pkl.load(fid)
		else:
			gaze_df = None
			for vid_id in tqdm(range(1, 75)):
				if vid_id in self._excluded_videos:
					continue
				if old:
					data_path = f'{dataset_path}/{vid_id:02d}/etg_samples.txt'
				else:
					data_path = f'{extra_annot_path}/gaze_data/{vid_id:02d}.txt'
				
				#print(f'Loading {data_path}...')
				#self._gaze_df_list.append(pd.read_csv(data_path, delimiter='\t', header=1))
				temp = pd.read_csv(data_path, delimiter=' ', header=0, na_filter=True)
				temp['vid_id'] = vid_id
				if gaze_df is None:
					gaze_df = temp
				else:
					gaze_df = pd.concat([gaze_df, temp], ignore_index=True)

			with open(cached_file, 'wb') as fid:
				pkl.dump(gaze_df, fid)
		return gaze_df

	def get_valid_gt_frames(self, gt='old'):
		'''
		get a list of valid (with non-zero gt) frames
		dataset: old (non-zero gt in old DReyeve gt)
				 new (non-zero gt in new DReyeVE gt)
				 old+new (non-zero gt in both old and new DReyeVE gt)
		'''

		empty_df = self._empty_df
		if gt == 'old':
			valid_frames_df = empty_df[empty_df['old_gt_status'] == 1][['vid_id', 'frame_id']]
		elif gt == 'new':
			valid_frames_df = empty_df[empty_df['new_gt_status'] == 1][['vid_id', 'frame_id']]
		elif gt in ['old+new', 'new+old']:
			valid_frames_df = empty_df[(empty_df['old_gt_status'] == 1) & (empty_df['new_gt_status'] == 1)]
		else:
			raise ValueError(f'Error: gt type "{gt}" is undefined!')
		return valid_frames_df


	def get_evaluation_frames(self, gt='new', obs_length=16):
		'''
			Get frames to compute evaluation metrics on
			For fair comparison of algorithms on 
		'''
		valid_frames_df = self.get_valid_gt_frames(gt=gt)[['vid_id', 'frame_id']]
		valid_frames_df = valid_frames_df[(valid_frames_df['frame_id'] > obs_length) & (valid_frames_df['vid_id'] > 37)]
		no_uturn_frames_df = self._vehicle_df[self._vehicle_df['lat action'] != 'U-turn'][['vid_id', 'frame_id']] # exclude frames with U-turns in them
		evaluation_frames_df = valid_frames_df.merge(no_uturn_frames_df, how='inner', left_on=['vid_id', 'frame_id'], right_on=['vid_id', 'frame_id'])[['vid_id', 'frame_id']]
		print('Excluded', 37*7500-len(evaluation_frames_df), 'frames')
		return evaluation_frames_df


	def get_intersection_frames(self):

		veh_df = self._vehicle_df.copy(deep=True)
		context_df = veh_df[['vid_id', 'frame_id', 'context']].dropna()
		# get context start frame before the intersection
		context_df[['intersection', 'priority', 'start_frame']] = context_df['context'].copy().str.split(';', n=2, expand=True)
		context_df.rename(columns={'frame_id':'end_frame'}, inplace=True)

		intersection_df = []
		for index, row in context_df.iterrows():
			start_frame = int(row['start_frame'])
			end_frame = int(row['end_frame'])
			vid_id = row['vid_id']
			intersection = row['intersection'].strip()
			priority = row['priority'].strip()
			for frame in range(start_frame, end_frame+1):
				intersection_df.append({'vid_id': vid_id, 'frame_id': frame, 'intersection': intersection, 'priority': priority})

		intersection_df = pd.DataFrame.from_dict(intersection_df)
		return context_df, intersection_df


	def label_action_frames(self, veh_df):

		veh_df['lat action'].fillna('drive straight', inplace=True)
		veh_df = veh_df.drop(veh_df[(veh_df['lat action'] == 'U-turn')].index)

		# https://journals.sagepub.com/doi/pdf/10.3141/2663-17
		# threshold for deceleration event in data is set to -0.4
		bins = [-10000, -0.4, 0.4, 10000]
		labels = ['decelerate', 'maintain', 'accelerate']
		veh_df['lon action'] = pd.cut(self._vehicle_df['acc'], bins=bins, labels=labels).astype(str)
		veh_df.loc[(veh_df['acc'] == 0) & (veh_df['speed'] < 1), 'lon action'] = 'stopped'	
		veh_df.loc[(veh_df['acc'] == 0) & (veh_df['speed'] < 1), 'lat action'] = 'stopped'	

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

	def get_action_frames(self, valid_frames_df=None):
		'''
		Get a dataframe with video ids and frame ids for the following actions:
		stopped = vehicle not moving
		no action = driving straight + maintain speed
		lat only = driving straight + decelerating
		lon only = turn left/right or lane change left/right
		lat/lon = intersection of lat only and lon only

		Args:
		action: name of the action
		valid_frames_df: list of valid video and frame ids (e.g. non-empty ground truth files)
		'''

		veh_df = self._vehicle_df.copy(deep=True)
		action_df = self.label_action_frames(veh_df)
		action_df = action_df[action_df['lat action'] != 'U-turn'] # remove frames with U-turns in them

		if valid_frames_df is not None: # remove empty
			action_df = action_df.merge(valid_frames_df, how='inner', right_on=['vid_id', 'frame_id'], left_on=['vid_id', 'frame_id'])

		return action_df



	def print_driver_action_stats(self):
		# number of actions
		# mean (std) action durations

		# print as a table (df.to_latex())
		# solution https://towardsdatascience.com/pandas-dataframe-group-by-consecutive-same-values-128913875dba
		veh_df = self._vehicle_df.copy(deep=True)
		
		veh_df = self.label_action_frames(veh_df)
		vid_ids = list(veh_df['vid_id'].unique()) 

		def get_action_stats(action_type, veh_df):

			prev_vid = -1
			prev_frame_id = -1

			action_counts = []
			prev_action = None
			record = None

			for idx, row in veh_df.iterrows():
				cur_vid = row['vid_id']
				cur_action = row[action_type]

				if (cur_action != prev_action) or (cur_vid != prev_vid):
					if record is not None:
						record['end_frame'] = prev_frame_id
						action_counts.append(record)
					record = {'vid_id': cur_vid, action_type: cur_action, 'num_frames': 1, 'start_frame': row['frame_id']}
				else:
					record['num_frames'] += 1

				prev_action = cur_action
				prev_vid = cur_vid
				prev_frame_id = row['frame_id']

			action_counts.append(record)
			
			action_counts_df = pd.DataFrame.from_dict(action_counts)
			#action_counts_df.to_excel(f'dreyeve_{action_type}_counts.xlsx')
			temp = action_counts_df[[action_type, 'num_frames']].groupby(action_type)
			action_stats_df = pd.concat([temp.count(), temp.sum(), temp.sum()/len(veh_df)*100, temp.mean(), temp.std()], axis=1)
			action_stats_df.columns = ['count', 'sum', 'perc', 'mean', 'std']

			return action_stats_df


		lat_action_stats_df = get_action_stats('lat action', veh_df)
		lon_action_stats_df = get_action_stats('lon action', veh_df)


		print(lon_action_stats_df.round(2).sort_values(by=['perc'], ascending=False).to_string())
		print(lat_action_stats_df.round(2).sort_values(by=['perc'], ascending=False).to_string())
		print(lon_action_stats_df[['count', 'mean', 'std', 'perc']].round(2).sort_values(by=['perc'], ascending=False).to_latex())
		print(lat_action_stats_df[['count', 'mean', 'std', 'perc']].round(2).sort_values(by=['perc'], ascending=False).to_latex())
		
		excel_filename = 'dreyeve_task_action.xlsx'
		
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
		inters_df[['inters_type', 'priority', 'start_frame']] = inters_df['context'].str.split(';', expand=True)
		inters_df = inters_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
		inters_df['duration'] = inters_df['frame_id']-inters_df['start_frame'].astype(int)

		temp = inters_df.groupby(['inters_type'])
		inters_stats_df = pd.concat([temp['duration'].count(), temp['duration'].mean(), temp['duration'].std(), temp['duration'].sum()/tot_frames*100], axis=1)
		inters_stats_df.columns = ['count', 'mean', 'std', 'perc']

		temp = inters_df[inters_df['vid_id'] < 38].groupby(['inters_type'])
		inters_stats_train_df = pd.concat([temp['duration'].count(), temp['duration'].mean(), temp['duration'].std(), temp['duration'].sum()/tot_frames*100], axis=1)
		inters_stats_train_df.columns = ['count', 'mean', 'std', 'perc']

		inters_stats_df = inters_df.groupby(['inters_type', 'priority']).count()
		
		excel_filename = 'dreyeve_task_action.xlsx'

		if os.path.exists(excel_filename):
			writer = pd.ExcelWriter(excel_filename, engine='openpyxl', mode='a', if_sheet_exists='replace')
			book = writer.book
			book.create_sheet('context')
		else:
			writer = pd.ExcelWriter(excel_filename, engine='openpyxl')
		
		inters_stats_df.to_excel(writer, sheet_name='context')
		writer.close()



if __name__ == '__main__':
	ds = DReyeVEUtils(cached=True)
	fire.Fire(ds)