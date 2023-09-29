# MAAD data utilities

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
from scipy.interpolate import interp1d
from tabulate import tabulate

extra_annot_path = os.path.join(os.environ['EXTRA_ANNOT_PATH'], 'MAAD')
dataset_path = os.environ['MAAD_PATH']
dreyeve_path = os.environ['DREYEVE_PATH']

class MAADUtils():
	def __init__(self, cached=True):
		self._vehicle_df = None
		self._excluded_videos = [6] # does not have accurate vehicle data
		self._gaze_df = None
		self._vehicle_df = None
		self._gaze_df = self.load_gaze_data(cached=cached)
		
		self.num_frames = 7500
		self.img_h = 1080
		self.img_w = 1920

		self.load_vehicle_data(cached=cached)
		
	def print_dataset_stats(self):
		"""
		number of videos
		number of frames
		number of valid videos
		number of valid frames
		number of eye-tracking data points
		"""

		num_subjects = len(self._gaze_df['subj_id'].drop_duplicates())
		num_conditions = len(self._gaze_df['condition'].drop_duplicates())
		vid_id_df = self._gaze_df[['vid_id', 'condition']].drop_duplicates()
		frame_id_df = self._gaze_df[['vid_id', 'condition', 'frame_gar']].drop_duplicates()
		subj_id_df = self._gaze_df[['vid_id', 'condition', 'subj_id']].drop_duplicates()
		num_videos = len(vid_id_df)
		num_frames = len(frame_id_df)


		num_subjects_per_video = subj_id_df.groupby(by=['vid_id', 'condition']).count()['subj_id'].to_frame().reset_index().groupby(['condition', 'subj_id']).count().reset_index()
		num_subjects_per_video.rename({'subj_id':'num_subjects', 'vid_id': 'num_videos'})
		print(num_subjects_per_video)
		

		vid_length = frame_id_df.groupby(by=['vid_id', 'condition']).count()['frame_gar']
		vid_length_mean = vid_length.mean()
		vid_length_std = vid_length.std()


		table = [['# subjects', num_subjects],
				 ['# conditions', num_conditions],
				 ['# videos', num_videos],
				 ['# frames', num_frames],
				 ['Video length (frames)', f'{vid_length_mean:0.2f}({vid_length_std:0.2f})'],
				 ['# subjects per video', f'{num_subjects_per_video["subj_id"].mean():0.2f}', f'({num_subjects_per_video["subj_id"].std():0.2f})']
				]
		print(tabulate(table))

	def get_video_attributes(self):
		video_attributes = []

		path_to_file = join(dreyeve_path, 'dr(eye)ve_design.txt')
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
				data_path = f'{extra_annot_path.replace("MAAD", "DReyeVE")}/vehicle_data/{vid_id:02d}.xlsx'
				#print(f'Loading {data_path}...')
				veh_df = pd.read_excel(data_path)
				veh_df['vid_id'] = vid_id
				if self._vehicle_df is None:
					self._vehicle_df = veh_df
				else:
					self._vehicle_df = pd.concat([self._vehicle_df, veh_df], ignore_index=True)
				#self._vehicle_df_list.append(pd.read_excel(data_path))
				if any(['Unnamed' in x for x in self._vehicle_df.columns]):
					print(vid_id)
			with open(cached_file, 'wb') as fid:
				pkl.dump(self._vehicle_df, fid)

	def convert_gaze_data(self):
		""" 
		Convert MAAD data from the original dict format to xlsx files
		Update ETG frames for MAAD using new DReyeVE annotations 
		"""
		
		att_aware_data_fname = os.path.join(dataset_path, 'att_aware_data_full/all_videos_subjects_tasks_gaze_data_dict.pkl')
		with open(att_aware_data_fname, 'rb') as fid:
			att_aware_data = pkl.load(fid)
		
		# the structure of the att_aware_data
		# att_aware_data[0] is a nested ordered dict with the following keys
		#	video_id (int)
		#		subj_id (int)
		#			condition (str)
		#				subj_gaze_data (pandas dataframe)

		dreyeve_gaze_dir = os.path.join(extra_annot_path.replace('MAAD', 'DReyeVE'), 'gaze_data')
		
		for vid_id, video_dict in tqdm(att_aware_data[0].items()):
			dreyeve_gaze_df = pd.read_csv(os.path.join(dreyeve_gaze_dir, f'{vid_id:02}.txt'), delimiter=' ', header=0, na_filter=False)
			first_row = dreyeve_gaze_df.iloc[0][['frame_etg', 'frame_gar']].values
			last_row = dreyeve_gaze_df.iloc[-1][['frame_etg', 'frame_gar']].values
			# grab a range of etg and gar frames from DReyeVE
			# and generate a set of etg frames for gar frames 0-7500
			num_points = last_row[1] - first_row[1]
			y = np.linspace(first_row[0], last_row[0], num=num_points)
			x = np.linspace(first_row[1], last_row[1], num=num_points)
			f = interp1d(x, y, fill_value='extrapolate')

			xnew = [round(x) for x in np.linspace(0, 7500, num=7500)]
			ynew = [round(x) for x in f(xnew)]

			index_df = pd.DataFrame({'frame_etg': ynew, 'frame_gar': xnew})			

			for subj_id, subj_dict in video_dict.items():
				for condition, subj_data in subj_dict.items():
					
					save_dir = os.path.join(extra_annot_path, 'gaze_data1', condition)
					os.makedirs(save_dir, exist_ok=True) 

					maad_gaze_df = subj_data.merge(index_df, how='outer', on='frame_gar')
					maad_gaze_df = maad_gaze_df.drop('frame_etg_x', axis=1)
					maad_gaze_df = maad_gaze_df.rename(columns={'frame_etg_y': 'frame_etg'})
					maad_gaze_df = maad_gaze_df[['frame_etg', 'frame_gar', 'X', 'Y', 'event_type', 'code']]
					maad_gaze_df.to_excel(os.path.join(save_dir, f'{vid_id:02d}_{subj_id:02d}.xlsx') , index=False, na_rep='n/a')

	def load_gaze_data(self, cached=True, old=False):
		print('-> Loading gaze data...')
		cached_file = 'cache/maad_gaze_data.pkl'

		if cached and os.path.exists(cached_file):
			with open(cached_file, 'rb') as fid:
				gaze_df = pkl.load(fid)
		else:
			gaze_df = None
			gaze_dir = os.path.join(extra_annot_path, 'gaze_data')
			exp_conditions = os.listdir(gaze_dir)
			for exp_condition in exp_conditions:
				gaze_files = sorted([x for x in os.listdir(os.path.join(gaze_dir, exp_condition)) if x.endswith('.xlsx')])
				for gaze_file_name in tqdm(gaze_files, desc=exp_condition):
					vid_id, subj_id = [int(x) for x in gaze_file_name.replace('.xlsx', '').split('_')]
					temp = pd.read_excel(os.path.join(gaze_dir, exp_condition, gaze_file_name), na_filter=False)
					temp['condition'] = exp_condition
					temp['vid_id'] = vid_id
					temp['subj_id'] = subj_id
					if gaze_df is None:
						gaze_df = temp
					else:
						gaze_df = pd.concat([gaze_df, temp], ignore_index=True)

			with open(cached_file, 'wb') as fid:
				pkl.dump(gaze_df, fid)
		return gaze_df


	def plot_gaze_stats(self, exp_condition=None):
		# gaze distribution sunburst diagram
		# errors, blinks, saccades, in-vehicle gaze, scene, out-of-bounds
		# TODO: plot fine-grained categories of gaze
		gaze_df = self._gaze_df.copy(deep=True)	

		if exp_condition is not None:
			gaze_df = gaze_df[gaze_df['condition']==exp_condition]


		event_types = gaze_df[['frame_gar', 'event_type']].groupby(['event_type'], as_index=False).count()
		#locations = gaze_df[['frame_gar', 'loc']].groupby(['loc']).count()

		#event_type_loc = gaze_df[['frame_gar', 'event_type', 'loc']].groupby(['event_type', 'loc'], as_index=False).count()
		#event_type_loc['perc'] = event_type_loc['frame_gar']/event_type_loc['frame_gar'].sum()*100

		#event_type_loc = event_type_loc.replace('NA', np.NaN)

		fig = px.pie(event_types, 
							values='frame_gar', 
							names='event_type')
		fig.update_traces(textinfo="label+percent")
		fig.update_layout(autosize=False, 
							height=500, 
							width=500, 
							paper_bgcolor="rgba(0,0,0,0)")
							# plot_bgcolor="rgba(0,0,0,0)")
		fig.show()
		fig.write_image('images/maad_gaze_types_sunburst.pdf')



if __name__ == '__main__':
	ds = MAADUtils(cached=True)
	fire.Fire(ds)