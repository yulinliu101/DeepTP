# -*- coding: utf-8 -*-
# @Author: liuyulin
# @Date:   2018-08-27 21:41:43
# @Last Modified by:   liuyulin
# @Last Modified time: 2018-08-28 18:33:13

import numpy as np
import pickle
from utils import downsample_track_data, baseline_time, GetAzimuth

class flight_track_feature_generator:
    def __init__(self, 
                 flight_track_dir,
                 flight_plan_dir,
                 flight_plan_util_dir,
                 wind_data_rootdir,
                 grbs_common_info_dir,
                 grbs_lvl_dict_dir,
                 grbs_smallgrid_kdtree_dir,
                 ncwf_arr_dir,
                 ncwf_alt_dict_dir,
                 **kwargs):

        self.flight_track_dir = flight_track_dir
        self.flight_plan_dir = flight_plan_dir
        self.flight_plan_util_dir = flight_plan_util_dir
        self.wind_data_rootdir = wind_data_rootdir
        self.grbs_common_info_dir = grbs_common_info_dir
        self.grbs_lvl_dict_dir = grbs_lvl_dict_dir
        self.grbs_smallgrid_kdtree_dir = grbs_smallgrid_kdtree_dir
        self.ncwf_arr_dir = ncwf_arr_dir
        self.ncwf_alt_dict_dir = ncwf_alt_dict_dir

        self.downsamp_rate_ft = kwargs.get('downsamp_rate_ft', 2)
        self.downsamp_rate_fp = kwargs.get('downsamp_rate_fp', 1.05)
        
    def data_loader(self):
        """
        load all datasets
        """
        print('================ Load wind/ tempr info =================')
        wind_fname_list = os.listdir(self.wind_data_rootdir)
        try:
            grbs_common_info = np.load(self.grbs_common_info_dir)
            # basegrid_lat = grbs_common_info['basegrid_lat']
            # basegrid_lon = grbs_common_info['basegrid_lon']
            # basegrid = grbs_common_info['basegrid']
            smallgrid = grbs_common_info['smallgrid']
            # small_idx = grbs_common_info['small_idx']
            lvls = grbs_common_info['levels']
        except:
            print('grbs common info not loaded')

        try:
            with open(self.grbs_lvl_dict_dir, 'rb') as pfile:
                lvls_dict = pickle.load(pfile)
        except:
            lvls_dict = {}
            i = 0
            for lvl in lvls:
                lvls_dict[lvl] = i
                i += 1
            with open(self.grbs_lvl_dict_dir, 'wb') as pfile:
                pickle.dump(lvls_dict, pfile)
        try:
            with open(self.grbs_smallgrid_kdtree_dir, 'rb') as pfile:
                smallgrid_tree = pickle.load(pfile)
        except:
            from scipy.spatial import cKDTree
            smallgrid_tree = cKDTree(smallgrid)
            
            with open(self.grbs_smallgrid_kdtree_dir, 'wb') as pfile:
                pickle.dump(smallgrid_tree, pfile)

        print('================ Load ncwf weather info =================')
        try:
            wx_pointer = np.load(self.ncwf_arr_dir)
            ncwf_arr = wx_pointer['ncwf_arr']
            start_time = wx_pointer['start_time']
            unique_alt = wx_pointer['unique_alt']
        except:
            print('NCWF weather data not loaded')
        try:
            with open(self.ncwf_alt_dict_dir, 'rb') as pfile:
                alt_dict = pickle.load(pfile)
        except:
            alt_dict = {}
            j = 0
            for lvl in unique_alt:
                alt_dict[lvl] = j
                j += 1
            with open(self.ncwf_alt_dict_dir, 'wb') as pfile:
                pickle.dump(alt_dict, pfile)


        print('================ Load flight track info =================')
        downsamp_flight_tracks, \
         downsamp_flight_plans, \
          flight_plans, \
           flight_plans_util, \
            flight_tracks = downsample_track_data(path_to_fp = self.flight_plan_dir,
                                                  path_to_fp_util = self.flight_plan_util_dir,
                                                  path_to_track = self.flight_track_dir,
                                                  downsamp_rate_ft = self.downsamp_rate_ft,
                                                  downsamp_rate_fp = self.downsamp_rate_fp)

        print('================ Datasets have been loaded into memory =================')
        return wind_fname_list, \
                smallgrid, \
                 lvls, \
                  lvls_dict, \
                   smallgrid_tree, \
                    ncwf_arr, \
                     start_time, \
                      unique_alt, \
                       alt_dict, \
                        downsamp_flight_tracks, \
                         downsamp_flight_plans, \
                          flight_plans, \
                           flight_plans_util, \
                            flight_tracks
    
    def flight_track_preprocess(self, flight_tracks):
        """
        decode time column (already done by function 'downsample_track_data')
        add time diff column
        add azimuth column (radian)
        add altitude to wind lvl column
        add altitude to wx alt column
        add wind fname column
        add wx fname column
        """

        flight_tracks['Elap_Time_Diff'] = flight_tracks.Elap_Time.apply(lambda x: (x - baseline_time).total_seconds())
        query_body = flight_tracks['Elap_Time_Diff'].values

        flight_tracks['azimuth'] = GetAzimuth(flight_tracks) * np.pi/180
        flight_tracks['levels'] = flight_tracks['Alt'].apply(lambda x: proxilvl(x*100, WindClass.lvls))
        flight_tracks['wx_alt'] = flight_tracks['Alt']//10
        





        return

    def feature_grid_generator(self):
        """
        generate grid for all track points in a batch
        """
        return

    def feature_to_georef_grid_mapper(self):
        """
        map feature grid to geo reference grid
        use kd tree
        """
        return

    def feature_arr_generator(self):
        """
        map feature grid to real value feature cube
        return numpy array (tensor)
        """
        return















test_wind_npz = np.load('../../DATA/filtered_weather_data/namanl_small_npz/namanl_218_20130101_0000_000.npz')
grbs_common_info = np.load('/media/storage/DATA/filtered_weather_data/grbs_common_info.npz')
basegrid_lat = grbs_common_info['basegrid_lat']
basegrid_lon = grbs_common_info['basegrid_lon']
basegrid = grbs_common_info['basegrid']
smallgrid = grbs_common_info['smallgrid']
small_idx = grbs_common_info['small_idx']
lvls = grbs_common_info['levels']
lvls_dict = {}
i = 0
for lvl in lvls:
    lvls_dict[lvl] = i
    i += 1

with open('/media/storage/DATA/filtered_weather_data/grbs_level_common_info.pkl', 'rb') as pfile:
    lvls_dict = pickle.load(pfile)
with open('/media/storage/DATA/filtered_weather_data/grbs_smallgrid_kdtree.pkl', 'rb') as pfile:
    smallgrid_tree = pickle.load(pfile)



wx_pointer = np.load('../../DATA/NCWF/gridded_storm.npz')
ncwf_arr = wx_pointer['ncwf_arr']
start_time = wx_pointer['start_time']
unique_alt = wx_pointer['unique_alt']
with open('../../DATA/NCWF/alt_dict.pkl', 'rb') as pfile:
    alt_dict = pickle.load(pfile)