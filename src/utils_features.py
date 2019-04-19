# -*- coding: utf-8 -*-
# @Author: liuyulin
# @Date:   2018-08-27 21:41:43
# @Last Modified by:   liuyulin
# @Last Modified time: 2018-10-14 15:41:02

import os
import numpy as np
import pickle
from scipy.spatial import cKDTree
import re
import datetime
import time
from utils import downsample_track_data, baseline_time, GetAzimuth, proxilvl, parser, create_grid_orient

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
                 downsample = True,
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

        self.downsample = downsample
        if self.downsample:
            self.downsamp_rate_ft = kwargs.get('downsamp_rate_ft', 2)
            self.downsamp_rate_fp = kwargs.get('downsamp_rate_fp', 1.05)
        self.load_ncwf_arr = kwargs.get('load_ncwf_arr', True)

        self.wind_fname_list, \
          self.lvls, \
           self.lvls_dict, \
            self.smallgrid_tree, \
             self.ncwf_wx_arr, \
              self.start_time, \
               self.wx_unique_alt, \
                self.wx_alt_dict, \
                 self.downsamp_flight_tracks, \
                  self.downsamp_flight_plans, \
                   self.ori_flight_plans, \
                    self.flight_plans_util, \
                     self.ori_flight_tracks = self.data_loader()
    def __str__(self):
        return 'flight track feature generator'
        
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
            smallgrid = grbs_common_info['smallgrid']
            smallgrid_tree = cKDTree(smallgrid)
            
            with open(self.grbs_smallgrid_kdtree_dir, 'wb') as pfile:
                pickle.dump(smallgrid_tree, pfile)

        print('================ Load ncwf weather info =================')
        try:
            wx_pointer = np.load(self.ncwf_arr_dir)
            if self.load_ncwf_arr:
                ncwf_wx_arr = wx_pointer['ncwf_arr']
            else:
                ncwf_wx_arr = None
                print('NCWF weather data array not loaded!')
            # ncwf_arr = None
            start_time = wx_pointer['start_time']
            wx_unique_alt = wx_pointer['unique_alt']
        except:
            print('NCWF weather data not loaded')
        try:
            with open(self.ncwf_alt_dict_dir, 'rb') as pfile:
                wx_alt_dict = pickle.load(pfile)
        except:
            wx_alt_dict = {}
            j = 0
            for lvl in wx_unique_alt:
                wx_alt_dict[lvl] = j
                j += 1
            with open(self.ncwf_alt_dict_dir, 'wb') as pfile:
                pickle.dump(wx_alt_dict, pfile)

        print('================ Load flight track info =================')
        if self.downsample:
            print('Downsampling...')
            downsamp_flight_tracks, \
             downsamp_flight_plans, \
              flight_plans, \
               flight_plans_util, \
                flight_tracks = downsample_track_data(path_to_fp = self.flight_plan_dir,
                                                      path_to_fp_util = self.flight_plan_util_dir,
                                                      path_to_track = self.flight_track_dir,
                                                      downsamp_rate_ft = self.downsamp_rate_ft,
                                                      downsamp_rate_fp = self.downsamp_rate_fp)
        else:
            print('Original...')
            import pandas as pd
            flight_plans = pd.read_csv(self.flight_plan_dir)
            flight_plans_util = pd.read_csv(self.flight_plan_util_dir)
            flight_tracks = pd.read_csv(self.flight_track_dir)
            flight_tracks['Elap_Time'] = pd.to_datetime(flight_tracks['Elap_Time'], errors = 'coerce')

            downsamp_flight_tracks = None
            downsamp_flight_plans = None


        print('================ Datasets have been loaded into memory =================')
        return wind_fname_list, \
                 lvls, \
                  lvls_dict, \
                   smallgrid_tree, \
                    ncwf_wx_arr, \
                     start_time, \
                      wx_unique_alt, \
                       wx_alt_dict, \
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
        print('================ PREPROCESSING FLIGHT TRACKS =================\n')
        print('================ decode time info =================')
        flight_tracks.loc[:, 'Elap_Time_Diff'] = flight_tracks.Elap_Time.apply(lambda x: (x - baseline_time).total_seconds())
        query_body = flight_tracks['Elap_Time_Diff'].values

        print('================ decode azimuth (back and forward, in radians) and altitude info =================')
        tmp_spd = flight_tracks['Speed'].shift(-1).values
        tmp_spd[-1] = 0
        flight_tracks.loc[:, 'Speed'] = tmp_spd
        flight_tracks.loc[:, 'azimuth'] = GetAzimuth(flight_tracks, course = False) * np.pi/180 # radian
        # azimuth is used to generate oriented feature cubes and corresponding grids
        flight_tracks.loc[:, 'course'] = GetAzimuth(flight_tracks, course = True) * np.pi/180 # radian
        flight_tracks.loc[:, 'levels'] = flight_tracks['Alt'].apply(lambda x: proxilvl(x*100, self.lvls_dict))
        flight_tracks.loc[:, 'wx_alt'] = flight_tracks['Alt']//10
        
        print('================ match weather name info =================')
        # match with wind/ temperature fname
        wind_query_idx, wind_valid_query, self.wind_time_objs, self.wind_ftime_tree = match_wind_fname(self.wind_fname_list, query_body, max_sec_bound = 3*3600)
        flight_tracks.loc[wind_valid_query, 'wind_fname'] = self.wind_time_objs[wind_query_idx[wind_valid_query], 0]
        flight_tracks.loc[~wind_valid_query, 'wind_fname'] = np.nan

        # match with ncwf idx
        wx_query_idx, wx_valid_query, wx_time_obj, self.wx_fname_hourly, self.wx_ftime_tree = match_ncwf_fname(self.start_time, query_body, max_sec_bound = 3600)
        flight_tracks.loc[wx_valid_query, 'wx_fname'] = self.wx_fname_hourly[wx_query_idx[wx_valid_query]]
        flight_tracks.loc[~wx_valid_query, 'wx_fname'] = np.nan
        flight_tracks.loc[wx_valid_query, 'wx_idx'] = wx_query_idx[wx_valid_query]
        flight_tracks.loc[~wx_valid_query, 'wx_idx'] = np.nan

        return flight_tracks

    def _feature_grid_generator(self,
                                theta_arr, 
                                shift_xleft = 0, 
                                shift_xright = 2, 
                                shift_yup = 1, 
                                shift_ydown = 1, 
                                nx = 20, 
                                ny = 20):
        """
        generate grid for all track points in a batch
        theta_arr = np.pi/2 - new_df.azimuth.values
        """
        block_grid = create_grid_orient(shift_xleft, shift_xright, shift_yup, shift_ydown, nx, ny, theta_arr)

        return block_grid

    def _feature_to_georef_grid_mapper(self, feature_grid):
        """
        map feature grid to geo reference grid
        use kd tree
        """
        _, query_idx = self.smallgrid_tree.query(feature_grid, p = 2)
        
        return query_idx


    def _generate_feature_cube(self, 
                               flight_tracks,
                               feature_grid_query_idx,
                               nx,
                               ny,
                               wx_alt_buffer = 20):
        """
        Given the flight track data (with agumented columns), generate wind and tempr cube for each track point
        use groupby function to speed up

        return a numpy array (tensor) with shape [None, 20, 20, 4]
        first layer: ncwf weather
        second layer: temperature
        third layer: u wind
        fourth layer: v wind
        """
        st = time.time()
        feature_cubes = np.zeros(shape = (feature_grid_query_idx.shape[0], nx, ny, 4), dtype = np.float32)

        #######################################################################################################
        groups = flight_tracks[['FID', 'wx_idx', 'wx_alt']].groupby(['wx_idx', 'wx_alt'])
        ng = groups.ngroups
        print('Extract ncwf convective weather from %d groups ...'%ng)
        for gpidx, gp in groups:
            # nan has been automatically dropped
            wx_alt_cover = self.wx_unique_alt[(self.wx_unique_alt >= (gpidx[1] - wx_alt_buffer)) & \
                                              (self.wx_unique_alt <= (gpidx[1] + wx_alt_buffer))]
            wx_alt_idxmin = self.wx_alt_dict[wx_alt_cover.min()]
            wx_alt_idxmax = self.wx_alt_dict[wx_alt_cover.max()] + 1
            wx_base = np.any(self.ncwf_wx_arr[int(gpidx[0]), wx_alt_idxmin: wx_alt_idxmax, :][:, feature_grid_query_idx[gp.index]], axis = 0).astype(np.float32).reshape(-1, nx, ny)
            feature_cubes[gp.index, :, :, 0] = wx_base
        print('Finished ncwf wx extraction!\n')

        #######################################################################################################
        groups = flight_tracks[['FID', 'wind_fname', 'levels']].groupby(['wind_fname', 'levels'])
        ng = groups.ngroups
        print('Extract wind/ temperature from %d groups ...'%ng)
        jj = -1
        for gpidx, gp in groups:
            jj += 1
            if jj % 1000 == 0:
                print('ELapsed time: %.2f seconds'%(time.time() - st))
                print('working on ', gpidx)
            # wind_npz = np.load(self.wind_data_rootdir + gpidx[0])
            # tmp_uwind = wind_npz['uwind']
            # tmp_vwind = wind_npz['vwind']
            # tmp_tempr = wind_npz['tempr']
            tmp_uwind, tmp_vwind, tmp_tempr = self._load_wind_low_memory(gpidx[0])
            uwind_base = tmp_uwind[self.lvls_dict[gpidx[1]]][feature_grid_query_idx[gp.index]].reshape(-1, nx,ny)
            vwind_base = tmp_vwind[self.lvls_dict[gpidx[1]]][feature_grid_query_idx[gp.index]].reshape(-1, nx,ny)
            tempr_base = tmp_tempr[self.lvls_dict[gpidx[1]]][feature_grid_query_idx[gp.index]].reshape(-1, nx,ny)

            feature_cubes[gp.index, :, :, 1] = tempr_base
            feature_cubes[gp.index, :, :, 2] = uwind_base
            feature_cubes[gp.index, :, :, 3] = vwind_base
            
        print('Finished wind/ temperature extraction!\n')
        
        return feature_cubes

    def _load_wind_low_memory(self, wind_fname):
        wind_npz = np.load(os.path.join(self.wind_data_rootdir, wind_fname))
        tmp_uwind = wind_npz['uwind']
        tmp_vwind = wind_npz['vwind']
        tmp_tempr = wind_npz['tempr']

        return tmp_uwind, tmp_vwind, tmp_tempr

    def feature_arr_generator(self, 
                              flight_tracks,
                              shift_xleft = 0, 
                              shift_xright = 2, 
                              shift_yup = 1, 
                              shift_ydown = 1, 
                              nx = 20, 
                              ny = 20):
        """
        map feature grid to real value feature cube
        return numpy array (tensor)
        """
        print('================ build feature grid =================')
        block_grid = self._feature_grid_generator(theta_arr = np.pi/2 - flight_tracks.azimuth.values,
                                                  shift_xleft = shift_xleft, 
                                                  shift_xright = shift_xright, 
                                                  shift_yup = shift_yup, 
                                                  shift_ydown = shift_ydown, 
                                                  nx = nx , 
                                                  ny = ny)
        feature_grid = (block_grid + flight_tracks[['Lon', 'Lat']].values.reshape(-1, 1, 2)).astype(np.float32)
        
        print('================ map feature grid to georef grid =================')
        query_idx = self._feature_to_georef_grid_mapper(feature_grid) # shape of [N_point, N_grid]; takes about 1 min for 1e5 points
        # query_idx is a one-to-one pointer to the flight_tracks index
        # i.e., row k of query_idx corresponds to row k of flight_tracks
        print('================ extract feature values from mapped georef grid =================\n')
        feature_cubes = self._generate_feature_cube(flight_tracks,
                                                   query_idx,
                                                   nx,
                                                   ny,
                                                   wx_alt_buffer = 20)

        return feature_cubes, feature_grid, query_idx


def match_wind_fname(wind_fname_list, query_body, max_sec_bound = 21960):
    # If there is wind file, then return the closest one within the max_sec_bound (in seconds)
    time_objs = []
    trash_holder = []
    wind_fname_list.sort()
    for item in wind_fname_list:
        if item.endswith('.npz'):
            time_string = re.findall('\d\d\d\d\d\d\d\d_\d\d\d\d_\d\d\d', item)[0]
            time_obj = parser.parse('%s %s'%(time_string[:8], time_string[9:13])) + \
                        datetime.timedelta(hours = int(time_string[-3:]))
            time_diff = (time_obj - baseline_time).total_seconds()
            if time_diff not in trash_holder:
                trash_holder.append(time_diff)
                time_objs.append([item, time_diff])
            else:
                time_objs.pop(-1)
                time_objs.append([item, time_diff])
        else:
            pass
        
    time_objs = np.array(time_objs, dtype=np.object)
    wind_ftime_tree = cKDTree(time_objs[:, -1].reshape(-1, 1))
    query_dist, query_index = wind_ftime_tree.query(query_body.reshape(-1,1), p = 1, distance_upper_bound = max_sec_bound)
    valid_query = query_index < time_objs.shape[0] # binary array
    return query_index, valid_query, time_objs, wind_ftime_tree

def match_ncwf_fname(start_time, query_body, max_sec_bound = 7200):
    # If there is any weather instance within the max_sec_bound, then return only the closest one (w.r.t. time)
    time_obj_wx = []
    time_diff_wx = []
    ncwf_fname_hourly = []
    for obj in start_time:
        tmp_fname = '%d_%s_%s_%s00Z.npz'%(obj[0], str(obj[1]).zfill(2), str(obj[2]).zfill(2), str(obj[3]).zfill(2))
        tmp_time = parser.parse('%d/%d/%d %d:00:00'%(obj[1], obj[2], obj[0], obj[3]))
        ncwf_fname_hourly.append(tmp_fname)
        time_obj_wx.append(tmp_time)
        time_diff_wx.append((tmp_time - baseline_time).total_seconds())
    
    ncwf_fname_hourly = np.array(ncwf_fname_hourly)
    time_diff_wx = np.array(time_diff_wx)
    time_obj_wx = np.array(time_obj_wx)
    wx_ftime_tree = cKDTree(time_diff_wx.reshape(-1, 1))

    query_dist, query_index = wx_ftime_tree.query(query_body.reshape(-1,1), p = 1, distance_upper_bound = max_sec_bound)
    valid_query = query_index < time_diff_wx.shape[0]
    return query_index, valid_query, time_obj_wx, ncwf_fname_hourly, wx_ftime_tree




# test_wind_npz = np.load('../../DATA/filtered_weather_data/namanl_small_npz/namanl_218_20130101_0000_000.npz')
# grbs_common_info = np.load('/media/storage/DATA/filtered_weather_data/grbs_common_info.npz')
# basegrid_lat = grbs_common_info['basegrid_lat']
# basegrid_lon = grbs_common_info['basegrid_lon']
# basegrid = grbs_common_info['basegrid']
# smallgrid = grbs_common_info['smallgrid']
# small_idx = grbs_common_info['small_idx']
# lvls = grbs_common_info['levels']
# lvls_dict = {}
# i = 0
# for lvl in lvls:
#     lvls_dict[lvl] = i
#     i += 1

# with open('/media/storage/DATA/filtered_weather_data/grbs_level_common_info.pkl', 'rb') as pfile:
#     lvls_dict = pickle.load(pfile)
# with open('/media/storage/DATA/filtered_weather_data/grbs_smallgrid_kdtree.pkl', 'rb') as pfile:
#     smallgrid_tree = pickle.load(pfile)



# wx_pointer = np.load('../../DATA/NCWF/gridded_storm.npz')
# ncwf_arr = wx_pointer['ncwf_arr']
# start_time = wx_pointer['start_time']
# unique_alt = wx_pointer['unique_alt']
# with open('../../DATA/NCWF/alt_dict.pkl', 'rb') as pfile:
#     alt_dict = pickle.load(pfile)
