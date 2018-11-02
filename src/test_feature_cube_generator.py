# -*- coding: utf-8 -*-
# @Author: liuyulin
# @Date:   2018-08-31 10:35:27
# @Last Modified by:   liuyulin
# @Last Modified time: 2018-10-28 16:58:58

from utils_features import flight_track_feature_generator
from utils import plot_fp_act, rotate_coord, plot_feature_grid
import numpy as np
import matplotlib.pyplot as plt


test_ffclass = flight_track_feature_generator(flight_track_dir = '/media/storage/DATA/DeepTPdata/New_IAHBOS2013.csv',
                                             flight_plan_dir = '/media/storage/DATA/DeepTPdata/cleaned_FP_tracks.CSV',
                                             flight_plan_util_dir = '/media/storage/DATA/DeepTPdata/IAH_BOS_Act_Flt_Trk_20130101_1231.CSV',
                                             wind_data_rootdir = '../../DATA/filtered_weather_data/namanl_small_npz/',
                                             grbs_common_info_dir = '/media/storage/DATA/filtered_weather_data/grbs_common_info.npz',
                                             grbs_lvl_dict_dir = '/media/storage/DATA/filtered_weather_data/grbs_level_common_info.pkl',
                                             grbs_smallgrid_kdtree_dir = '/media/storage/DATA/filtered_weather_data/grbs_smallgrid_kdtree.pkl',
                                             ncwf_arr_dir = '../../DATA/NCWF/gridded_storm.npz',
                                             ncwf_alt_dict_dir = '../../DATA/NCWF/alt_dict.pkl',
                                             load_ncwf_arr = False)

# downsample
assert test_ffclass.downsamp_flight_plans.shape[0] < test_ffclass.ori_flight_plans.shape[0]
assert test_ffclass.downsamp_flight_plans.shape[1] == test_ffclass.ori_flight_plans.shape[1]
assert test_ffclass.downsamp_flight_tracks.shape[0] == (test_ffclass.ori_flight_tracks.shape[0] - \
                                                         test_ffclass.ori_flight_tracks.FID.unique().shape[0]*2)//2 + \
                                                            test_ffclass.ori_flight_tracks.FID.unique().shape[0]*2
assert test_ffclass.downsamp_flight_tracks.shape[1] == test_ffclass.ori_flight_tracks.shape[1]
print('Downsamping checked!')

# azimuth
processed_flight_tracks = test_ffclass.flight_track_preprocess(test_ffclass.downsamp_flight_tracks)
assert np.less_equal(processed_flight_tracks.groupby('FID').head(1).azimuth.unique(), np.array([1e6]))
print(processed_flight_tracks.columns)
print('Azimuth checked!')

# feature grid

block_grid = test_ffclass._feature_grid_generator(theta_arr = np.pi/2 - processed_flight_tracks.azimuth.values,
                                                  shift_xleft = 0, 
                                                  shift_xright = 2, 
                                                  shift_yup = 1, 
                                                  shift_ydown = 1, 
                                                  nx = 20, 
                                                  ny = 20)
feature_grid = (block_grid + processed_flight_tracks[['Lon', 'Lat']].values.reshape(-1, 1, 2)).astype(np.float32)

assert feature_grid.shape == (processed_flight_tracks.shape[0], 400, 2)

for _ in range(10):
    idx = np.random.randint(0, processed_flight_tracks.shape[0])
    x = np.linspace(0, 2, 20)
    y = np.linspace(-1, 1, 20)
    xv, yv = np.meshgrid(x, y, sparse = False)
    grid_2d = np.vstack((xv.flatten(),yv.flatten()))

    test_grid = rotate_coord((feature_grid[idx] - processed_flight_tracks.loc[idx, ['Lon', 'Lat']].values.reshape(-1, 2)).T,
                             [-np.pi/2+processed_flight_tracks.loc[idx, 'azimuth']]).reshape(2, 400)

    assert np.linalg.norm(test_grid - grid_2d) <= 1e4
# visual inspection
plot_feature_grid(feature_grid[0:78], processed_flight_tracks.iloc[:78][['Lon', 'Lat']].values)
plt.show()
plot_feature_grid(feature_grid[-90:], processed_flight_tracks.iloc[-78:][['Lon', 'Lat']].values)
plt.show()
print('feature grid checked!')

# feature cubes
import sys
sys.path.insert(0, '../weather/')
from utils_cube import plot_daily_wx

import pickle

feature_cubes_pointer = np.load('../../DATA/DeepTP/feature_cubes.npz')
feature_cubes = feature_cubes_pointer['feature_cubes.npy']
feature_grid = feature_cubes_pointer['feature_grid']
feature_grid_qidx = feature_cubes_pointer['feature_grid_qidx']

print('feature cube shape ...')
print(processed_flight_tracks.shape)
print(feature_cubes.shape)
print(feature_grid.shape)
print(feature_grid_qidx.shape)

print('visual inspection on ncwf weather polygons')
ncwf_wx = pickle.load(open('../../DATA/NCWF/processed_bundle/storm_bundle_merge.pkl', 'rb'))
## weather should be good
idx = 51134
print(feature_grid[idx].shape)
tmp_dt = processed_flight_tracks.loc[idx, 'Elap_Time']
print(str(tmp_dt))
print(processed_flight_tracks.loc[idx, 'wx_fname'])
tmp_wx_idx = ncwf_wx[2].index([tmp_dt.year, tmp_dt.month, tmp_dt.day, tmp_dt.hour])
plt.imshow(feature_cubes[idx][:, :, 0], origin='lower')
bad_poly = plot_daily_wx(ncwf_wx[0][tmp_wx_idx-1:tmp_wx_idx+1], feature_grid = feature_grid[idx])
plt.show()

## wind and temperature should be good
print('visual inspection on nam wind and temperature data')

for _ in range(3):
    idx = np.random.randint(0, processed_flight_tracks.shape[0])
    tmp_dt = processed_flight_tracks.loc[idx, ['levels', 'wind_fname']]
    print(str(tmp_dt.wind_fname))
    wind_npz = np.load('../../DATA/filtered_weather_data/namanl_small_npz/' + tmp_dt.wind_fname)
    tmp_uwind = wind_npz['uwind']
    tmp_vwind = wind_npz['vwind']
    tmp_tempr = wind_npz['tempr']
    plt.figure(figsize = (10,6))
    plt.subplot(2, 3, 1)
    plt.imshow(tmp_tempr[test_ffclass.lvls_dict[tmp_dt.levels]][feature_grid_qidx[idx]].reshape(20, 20))
    plt.subplot(2, 3, 2)
    plt.imshow(tmp_uwind[test_ffclass.lvls_dict[tmp_dt.levels]][feature_grid_qidx[idx]].reshape(20, 20))
    plt.subplot(2, 3, 3)
    plt.imshow(tmp_vwind[test_ffclass.lvls_dict[tmp_dt.levels]][feature_grid_qidx[idx]].reshape(20, 20))
    plt.subplot(2, 3, 4)
    plt.imshow(feature_cubes[idx][:, :, 1])
    plt.subplot(2, 3, 5)
    plt.imshow(feature_cubes[idx][:, :, 2])
    plt.subplot(2, 3, 6)
    plt.imshow(feature_cubes[idx][:, :, 3])
    plt.show()