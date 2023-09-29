import os
import re
import math
import folium
import argparse
import datetime
import pickle
import json
import cv2
import numpy as np
import osmnx as ox
import selenium.webdriver
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from valhalla import Actor, get_config
from geopy.distance import geodesic as GD


def map_match(actor: Actor,
              df: pd.DataFrame) -> str:
    param = {
        "use_timestamps": True,
        "shortest": True,
        "shape_match": "walk_or_snap",
        "shape": df.to_dict(orient='records'),
        "costing": "auto",
        "format": "osrm",
        "directions_options": {
            "directions_type": "none"
        },
        "trace_options": {
            "search_radius": 5,
            "max_search_radius": 100,
            "gps_accuracy": 5,
            "breakage_distance": 2000,
            "turn_penalty_factor": 1
        },
        "costing_options": {
            "auto": {
                "country_crossing_penalty": 2000.0,
                "maneuver_penalty": 30
            }
        }
    }
    route = actor.trace_route(param)
    attrs = actor.trace_attributes(param)
    return route, attrs


class ProcessGPS:
    def __init__(self):
        self._gps_data = []
        self._map = None

    def setup_dataset(self):
        pass

    def load_gps_data(self, vid_id=None):
        pass

    def load_map(self):
        lat_list = self._raw_gps_df['lat'].dropna()
        lon_list = self._raw_gps_df['lon'].dropna()

        mid_path = int(len(lat_list)/2)
        s, w = [min([x for x in lat_list]), min([x for x in lon_list])]
        n, e = [max([x for x in lat_list]), max([x for x in lon_list])]

        off = self._off
        print(n+off, s-off, e+off, w-off)

        graph=ox.graph_from_bbox(n+off, s-off, e+off, w-off, network_type='drive', retain_all=True)
        self._map = ox.graph_to_gdfs(graph)


    def map_match_gps(self):
        try:
            config = get_config(tile_extract='valhalla/custom_files/valhalla_tiles.tar', verbose=True)
            actor = Actor(config)
            raw_gps_df = self._raw_gps_df.dropna()[['lat', 'lon']]
            route, attrs = map_match(actor, raw_gps_df)
            self._matched_gps_df = pd.DataFrame.from_dict(attrs['matched_points'])[['lat', 'lon']]
        except Exception as e:
            print('ERROR:', str(e), f'for {self._vid_id}')
            self._matched_gps_df = raw_gps_df

    def save_map_image(self, debug=True):
        '''
        Save rasterized map image so that 1px = 1m in the final result
        '''
        dpi = 100
        
        plt.style.use('dark_background')

        fig, ax = plt.subplots()
        
        nodes, edges = self._map
        edges.plot(ax=ax, linewidth=self._linewidth, edgecolor='white')

        ylim = ax.get_ylim()
        xlim = ax.get_xlim()

        h_m = GD([ylim[0], xlim[0]], [ylim[1], xlim[0]]).m
        w_m = GD([ylim[0], xlim[0]], [ylim[0], xlim[1]]).m


        fig.set_size_inches((w_m/dpi, h_m/dpi))
        plt.gca().set_position((0, 0, 1, 1))
        ax.set_axis_off()

        save_path = os.path.join(self._extra_annot_dir, 'route_maps', f'{self._vid_id:02d}.png')            

        plt.savefig(save_path, format='png', bbox_inches='tight', pad_inches=0, dpi=dpi)

        map_json_path = os.path.join(self._extra_annot_dir, 'route_maps', 'map_info.json')
        if os.path.exists(map_json_path):
            with open(map_json_path, 'r') as fid:
                map_dict = json.load(fid)
        else:
            map_dict = {}

        map_dict[self._vid_id] = {'xlim': xlim, 'ylim': ylim, 'h_m': h_m, 'w_m': w_m}


        with open(map_json_path, 'w') as fid:
            fid.write(json.dumps(map_dict, indent=4))

        if debug:

            map_img = cv2.imread(save_path)
            h_px, w_px, c = map_img.shape

            print(map_img.shape, 'm/px', h_m/map_img.shape[0], w_m/map_img.shape[1])

            orig_gps_df = self._raw_gps_df[['lat', 'lon']].dropna()
            matched_gps_df = self._matched_gps_df[['lat', 'lon']].dropna()

            orig_lat = h_px - ((orig_gps_df['lat'].values - ylim[0])*h_px)/(ylim[1]-ylim[0])
            orig_lon = ((orig_gps_df['lon'].values - xlim[0])*w_px)/(xlim[1]-xlim[0]) 

            matched_lat = h_px - ((matched_gps_df['lat'].values - ylim[0])*h_px)/(ylim[1]-ylim[0]) 
            matched_lon = ((matched_gps_df['lon'].values - xlim[0])*w_px)/(xlim[1]-xlim[0]) 

            for y, x in zip(matched_lat, matched_lon):
                cv2.circle(map_img, (int(x), int(y)), radius=2, color=(0,255,0), thickness=-1)
            
            for y, x in zip(orig_lat, orig_lon):
                cv2.circle(map_img, (int(x), int(y)), radius=2, color=(0,0,255), thickness=-1)

            cv2.imwrite(self._debug_map_filename, map_img)

        plt.close(fig)

    def save_map_html(self):

        lat_m = self._matched_gps_df['lat'].values
        lon_m = self._matched_gps_df['lon'].values
        
        lat_orig = self._raw_gps_df['lat'].values
        lon_orig = self._raw_gps_df['lon'].values

        mid_path = int(len(lat_m)/2)

        map = folium.Map(location=[lat_m[mid_path], lon_m[mid_path]], 
                        zoomlevel=19, zoom_control=True, max_zoom=25,
                        scrollWheelZoom=True, dragging=True, no_touch=False)

        i = self._step # plot every i-th point 
        sw = [min([x for x in lat_m]), min([x for x in lon_m[::i]])]
        ne = [max([x for x in lat_m]), max([x for x in lon_m[::i]])]

        for idx, (ltm, lnm, lto, lno) in enumerate(zip(lat_m, lon_m, lat_orig, lon_orig)):
            #if idx == 0:
                #folium.Marker(location=[ltm, lnm], icon=folium.Icon(color='blue'), popup='Start').add_to(map) # mark beginning of the route
            if not (np.isnan(ltm) or np.isnan(lnm)):
                folium.CircleMarker(location=[ltm, lnm], 
                                    popup=f'({idx}: {ltm}, {lnm})', 
                                    radius=5, weight=0.5, color='green').add_to(map) # mark beginning of the route
            if not (np.isnan(lto) or np.isnan(lno)):
                folium.CircleMarker(location=[lto, lno], 
                                popup=f'({idx}: {lto}, {lno})', 
                                radius=5, weight=0.5, color='red').add_to(map) # mark beginning of the route

        map.fit_bounds([sw, ne])
        map.save(self._map_filename)           

        return map        

    def save_gps_data(self):
        self._vehicle_df = pd.read_excel(self._gps_save_path)
        self._vehicle_df['lat_m'] = self._matched_gps_df['lat']
        self._vehicle_df['lon_m'] = self._matched_gps_df['lon']
        self._vehicle_df.to_excel(self._gps_save_path, columns=['frame', 'speed', 'course', 'lat', 'lon', 'lat_m', 'lon_m', 'lat action', 'acc', 'context'])


    def process_gps_data_single(self, vid_id):

        self.setup_paths(vid_id)

        ok = self.load_gps_data(vid_id=vid_id)
        if not ok:
            print(f'Skipping {vid_id}, no gps data!')
            return

        if os.path.exists(self._map_filename):
            print(f'Skipping {vid_id}, already processed {self._map_filename}!')
            return

        self.load_map()
        self.map_match_gps()
        self.interpolate_gps()
        self.save_gps_data()
        self.save_map_image() 
        self.save_map_html()       

    def process_gps_data(self, vid_id=-1):
        if vid_id < 0:
            for video_id in self._dataset_dict.keys():        
                self.process_gps_data_single(video_id)
        else:
            self.process_gps_data_single(vid_id)


class ProcessGPSDReyeVE(ProcessGPS):
    def __init__(self):
        
        ProcessGPS.__init__(self)

        self._dataset_dict = {}
        self._data_dir = os.environ['DREYEVE_PATH']
        self._extra_annot_dir = os.path.join(os.environ['EXTRA_ANNOT_PATH'], 'DReyeVE')
        self._raw_gps_df = None
        self._matched_gps_df = None
        self._map = None
        self._vid_id = None
        self._off = 0.01
        self._step = 30
        self._linewidth = 5
        self.setup_dataset()    

    def setup_paths(self, vid_id):   
        self._map_filename = f'{self._extra_annot_dir}/route_maps/debug/{vid_id:02d}.html' 
        self._debug_map_filename = os.path.join(self._extra_annot_dir, 'route_maps/debug', f'{vid_id:02d}_debug.png')
        self._map_image_filename = os.path.join(self._extra_annot_dir, 'route_maps', f'{vid_id:02d}.png')
        self._gps_save_path = os.path.join(self._extra_annot_dir, 'vehicle_data', f'{vid_id:02d}.xlsx')
    
    def setup_dataset(self):
        for vid_id in range(1, 75):
            video_dir = os.path.join(self._data_dir, f'{vid_id:02}')
            self._dataset_dict[vid_id] = {}
            self._dataset_dict[vid_id]['video_dir'] = video_dir
            self._dataset_dict[vid_id]['gps_data'] = os.path.join(video_dir, 'speed_course_coord.txt')

    def load_gps_data(self, vid_id):
        self._vid_id = vid_id
        print(f'Loading GPS data...{vid_id}')
        gps_data_file = self._dataset_dict[vid_id]['gps_data']
        header = ['frame', 'speed', 'course', 'lat', 'lon']
        self._raw_gps_df = pd.read_csv(gps_data_file, index_col=False, delimiter='\t', names=header)
        return True

    def interpolate_gps(self):
        self._matched_gps_df.index = self._raw_gps_df.dropna().index
        self._matched_gps_df = self._matched_gps_df.reindex(range(7500), fill_value=np.nan)
        self._matched_gps_df['lat'] = self._matched_gps_df['lat'].interpolate(limit_direction='both')
        self._matched_gps_df['lon'] = self._matched_gps_df['lon'].interpolate(limit_direction='both')


class ProcessGPSBDDA(ProcessGPS):
    def __init__(self):
        
        ProcessGPS.__init__(self)
        
        self._dataset_dict = {}
        self._data_dir = os.environ['BDDA_PATH']
        self._extra_annot_dir = os.path.join(os.environ['EXTRA_ANNOT_PATH'], 'BDD-A')
        self._raw_gps_df = None
        self._matched_gps_df = None
        self._map = None
        self._vid_id = None
        self._linewidth = 1
        self._off = 0.005 # offset in degrees for expanding the map beyond the coordinates of the trajectory
                          # needed to preserve road geometry when cropping the osm street graph
        self._step = 30 # how many points to skip when plotting
        self.setup_dataset()    

    def setup_paths(self, vid_id):   
        self._map_filename = f'{self._extra_annot_dir}/route_maps/debug/{vid_id}.html' 
        self._debug_map_filename = os.path.join(self._extra_annot_dir, 'route_maps/debug', f'{vid_id}_debug.png')
        self._map_image_filename = os.path.join(self._extra_annot_dir, 'route_maps', f'{vid_id}.png')
        self._gps_save_path = os.path.join(self._extra_annot_dir, 'vehicle_data', f'{vid_id}.xlsx')

    def setup_dataset(self):

        for data_type in ['test', 'training', 'validation']:
            data_type_dir = os.path.join(self._data_dir, data_type)
            vid_ids = os.listdir(os.path.join(data_type_dir, 'camera_videos'))
            vid_ids = [os.path.splitext(x)[0] for x in vid_ids] 
            
            for vid_id in vid_ids:            
                vid = int(vid_id)
                self._dataset_dict[vid] = {}
                self._dataset_dict[vid]['video_dir'] = data_type_dir
                self._dataset_dict[vid]['vehicle_file'] = os.path.join(data_type_dir, 'gps_jsons', vid_id+'.json')
                self._dataset_dict[vid]['num_frames'] = len(os.listdir(os.path.join(data_type_dir, 'camera_frames', str(vid_id))))


    def load_map(self):
        lat_list = self._raw_gps_df['lat'].dropna().values
        lon_list = self._raw_gps_df['lon'].dropna().values

        mid_path = int(len(lat_list)/2)
        s, w = [min([x for x in lat_list]), min([x for x in lon_list])]
        n, e = [max([x for x in lat_list]), max([x for x in lon_list])]

        off = self._off
        print(n+off, s-off, e+off, w-off)

        try:
            graph=ox.graph_from_point((lat_list[mid_path], lon_list[mid_path]), dist=500, network_type='drive', retain_all=True)
            self._map = ox.graph_to_gdfs(graph)
        except:
            graph=ox.graph_from_point((lat_list[mid_path], lon_list[mid_path]), dist=1500, network_type='drive', retain_all=True)
            self._map = ox.graph_to_gdfs(graph)

    def load_gps_data(self, vid_id):
        self._vid_id = vid_id

        if not os.path.exists(self._gps_save_path):
            return False

        print(f'Loading GPS data...{vid_id}')
        try:
            vehicle_json = self._dataset_dict[vid_id]['vehicle_file']

        except KeyError:
            return False


        gps_json = self._dataset_dict[vid_id]['vehicle_file']
        gps_data = []
        if os.path.exists(gps_json):
            with open(gps_json, 'r') as vf:
                vehicle_data = json.load(vf)

            startTime = vehicle_data['startTime'] # get start time
            endTime = startTime + int(self._dataset_dict[self._vid_id]['num_frames']/29*1000)

            # frame_timestamps = [int(x) for x in np.linspace(startTime, endTime, num=self._dataset_dict[vid_id]['num_frames'], endpoint=True)]


            # for frame_num, timestamp in enumerate(frame_timestamps):
            #     gps_data.append({'frame_num': frame_num,
            #                        'timestamp': timestamp, 
            #                        'speed': float('nan'),
            #                        'course': float('nan'),
            #                        'lat': float('nan'),
            #                        'lon': float('nan')})

            step = 29 # BDD-A framerate
            self._raw_gps_df = pd.DataFrame.from_dict(vehicle_data['locations']).sort_values('timestamp')
            self._raw_gps_df.index = pd.RangeIndex(start=0, stop=len(self._raw_gps_df.index)*step-1, step=step)
            self._raw_gps_df.rename(columns={'latitude':'lat', 'longitude':'lon'}, inplace=True)
        else:
            return False
        return True            

    def interpolate_gps(self):
        self._matched_gps_df.index = self._raw_gps_df.dropna().index
        self._matched_gps_df = self._matched_gps_df.reindex(range(self._dataset_dict[self._vid_id]['num_frames']), fill_value=np.nan)
        self._matched_gps_df['lat'] = self._matched_gps_df['lat'].interpolate(limit_direction='both')
        self._matched_gps_df['lon'] = self._matched_gps_df['lon'].interpolate(limit_direction='both')

        self._raw_gps_df = self._raw_gps_df.reindex(range(self._dataset_dict[self._vid_id]['num_frames']), fill_value=np.nan)
        self._raw_gps_df['lat'] = self._raw_gps_df['lat'].interpolate(limit_direction='both')
        self._raw_gps_df['lon'] = self._raw_gps_df['lon'].interpolate(limit_direction='both')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Clean-up and plot GPS data for DReyeVE and BDD-A')
    parser.add_argument('--dataset', help='DReyeVE or BDD-A', default='DReyeVE', type=str)
    parser.add_argument('--vid_id', help='Video id', default=-1, type=int)

    args = vars(parser.parse_args())

    if args['dataset'].lower() == 'dreyeve':
        p = ProcessGPSDReyeVE()
    elif args['dataset'].lower() == 'bdd-a':
        p = ProcessGPSBDDA()
    else:
        raise ValueError(f'ERROR: dataset {dataset} is not supported!')

    p.process_gps_data(vid_id=args['vid_id'])

