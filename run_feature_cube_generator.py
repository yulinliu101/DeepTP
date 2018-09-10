# -*- coding: utf-8 -*-
# @Author: liuyulin
# @Date:   2018-08-29 18:59:39
# @Last Modified by:   liuyulin
# @Last Modified time: 2018-08-30 09:42:16

from utils_features import *
from utils import *
import pandas as pd
import numpy as np
feature_cube_class = flight_track_feature_generator(flight_track_dir = '/media/storage/DATA/DeepTPdata/New_IAHBOS2013.csv',
                                                     flight_plan_dir = '/media/storage/DATA/DeepTPdata/cleaned_FP_tracks.CSV',
                                                     flight_plan_util_dir = '/media/storage/DATA/DeepTPdata/IAH_BOS_Act_Flt_Trk_20130101_1231.CSV',
                                                     wind_data_rootdir = '../../DATA/filtered_weather_data/namanl_small_npz/',
                                                     grbs_common_info_dir = '/media/storage/DATA/filtered_weather_data/grbs_common_info.npz',
                                                     grbs_lvl_dict_dir = '/media/storage/DATA/filtered_weather_data/grbs_level_common_info.pkl',
                                                     grbs_smallgrid_kdtree_dir = '/media/storage/DATA/filtered_weather_data/grbs_smallgrid_kdtree.pkl',
                                                     ncwf_arr_dir = '../../DATA/NCWF/gridded_storm.npz',
                                                     ncwf_alt_dict_dir = '../../DATA/NCWF/alt_dict.pkl')

flight_tracks = feature_cube_class.flight_track_preprocess(feature_cube_class.downsamp_flight_tracks)
flight_tracks.to_csv('../../DATA/DeepTP/processed_flight_tracks.csv', index = True)
feature_cube_class.downsamp_flight_plans.to_csv('../../DATA/DeepTP/processed_flight_plans.csv', index = False)



feature_cubes, feature_grid, query_idx = feature_cube_class.feature_arr_generator(flight_tracks,
                                                                                  shift_xleft = 0, 
                                                                                  shift_xright = 2, 
                                                                                  shift_yup = 1, 
                                                                                  shift_ydown = 1, 
                                                                                  nx = 20, 
                                                                                  ny = 20)
np.savez_compressed('../../DATA/DeepTP/feature_cubes.npz', feature_cubes = feature_cubes, feature_grid = feature_grid, feature_grid_qidx = query_idx)