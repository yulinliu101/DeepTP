import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os
try:
    import cPickle as pickle
except:
    import pickle
from utils import g, baseline_time

"""
Difference between datasets.py
Change the state variables from [lat (deg), lon (deg), alt (FL), spd (nmi/sec), course (rad)]
to: [lat (deg), lon (deg), alt (FL), lat_dot (deg/sec), lon_dot (deg/sec)]

"""

class DatasetEncoderDecoder:
    def __init__(self,
                 actual_track_datapath,
                 flight_plan_datapath,
                 flight_plan_utilize_datapath,
                 feature_cubes_datapath,
                 shuffle_or_not = True,
                 split = True,
                 batch_size = 128,
                 **kwargs):
        print("State variables as in [Lat, lon, alt, cumDT, lat_spd, lon_spd]")
        self.actual_track_datapath = actual_track_datapath
        self.flight_plan_datapath = flight_plan_datapath
        self.flight_plan_utilize_datapath = flight_plan_utilize_datapath
        self.feature_cubes_datapath = feature_cubes_datapath
        self.shuffle_or_not = shuffle_or_not
        self.split = split
        self.batch_size = batch_size

        self.dep_lat = kwargs.get('dep_lat', 29.98333333)
        self.dep_lon = kwargs.get('dep_lon', -95.33333333)
        self.arr_lat = kwargs.get('arr_lat', 42.3666666667)
        self.arr_lon = kwargs.get('arr_lon', -70.9666666667)
        self.time_dim = kwargs.get('time_dim', False)

        self.direct_course = kwargs.get('direct_course', g.inv(self.dep_lon, self.dep_lat, self.arr_lon, self.arr_lat)[0]* np.pi/180)
        self.idx = kwargs.get('idx', 0)

        self.all_tracks, \
         self.all_targets, \
          self.all_targets_end, \
           self.all_targets_end_neg, \
            self.all_seq_lens, \
             self.data_mean, \
              self.data_std, \
               self.all_FP_tracks, \
                self.all_seq_lens_FP, \
                 self.FP_mean, \
                  self.FP_std, \
                   self.tracks_time_id_info = self.load_track_data()

        self.feature_cubes, self.feature_cubes_mean, self.feature_cubes_std = self.load_feature_cubes()
        self.feature_cubes = np.split(self.feature_cubes, np.cumsum(self.all_seq_lens))[:-1]

        if self.shuffle_or_not:
            self.all_tracks, \
             self.all_targets,\
              self.all_targets_end,\
               self.all_targets_end_neg,\
                self.all_seq_lens, \
                 self.all_FP_tracks, \
                  self.all_seq_lens_FP, \
                   self.feature_cubes,\
                    self.tracks_time_id_info = shuffle(self.all_tracks, 
                                                       self.all_targets,
                                                       self.all_targets_end,
                                                       self.all_targets_end_neg,
                                                       self.all_seq_lens, 
                                                       self.all_FP_tracks, 
                                                       self.all_seq_lens_FP, 
                                                       self.feature_cubes,
                                                       self.tracks_time_id_info,
                                                       random_state = 101)

        if self.split:
            self.train_tracks, \
             self.dev_tracks, \
              self.train_targets, \
               self.dev_targets, \
                self.train_targets_end, \
                 self.dev_targets_end, \
                  self.train_targets_end_neg, \
                   self.dev_targets_end_neg, \
                    self.train_seq_lens, \
                     self.dev_seq_lens, \
                      self.train_FP_tracks, \
                       self.dev_FP_tracks, \
                        self.train_seq_lens_FP, \
                         self.dev_seq_lens_FP, \
                          self.train_feature_cubes, \
                           self.dev_feature_cubes, \
                            self.train_tracks_time_id_info, \
                             self.dev_tracks_time_id_info = train_test_split(self.all_tracks,
                                                                             self.all_targets,
                                                                             self.all_targets_end,
                                                                             self.all_targets_end_neg, 
                                                                             self.all_seq_lens, 
                                                                             self.all_FP_tracks,
                                                                             self.all_seq_lens_FP,
                                                                             self.feature_cubes,
                                                                             self.tracks_time_id_info,
                                                                             random_state = 101, 
                                                                             train_size = 0.8,
                                                                             test_size = None)

        self.train_tracks = _pad(self.train_tracks, self.train_seq_lens)
        self.train_targets = _pad(self.train_targets, self.train_seq_lens) 
        self.train_targets_end = _pad(self.train_targets_end, self.train_seq_lens) 
        self.train_targets_end_neg = _pad(self.train_targets_end_neg, self.train_seq_lens) 
        
        self.train_feature_cubes = _pad(self.train_feature_cubes, self.train_seq_lens)
        self.n_train_data_set = self.train_tracks.shape[0]
    def __str__(self):
        return 'Dataset Class to Conduct Training Procedure'

    def _calc_latlon_spd(self, track_dataframe):
        CenterTraj = track_dataframe[['FID', 'Lat', 'Lon', 'DT']]
        # CenterTraj.loc[:, 'azimuth'] = last_pnt
        tmp_df = CenterTraj.shift(-1)
        latlon_spd = np.divide((tmp_df[['Lat', 'Lon']].values - CenterTraj[['Lat', 'Lon']].values), tmp_df.DT.values.reshape(-1,1))
        tmp_tail_idx = CenterTraj.groupby("FID")['Lat'].tail(1).index
        latlon_spd[tmp_tail_idx, :] = 0.
        latlon_spd[np.isnan(latlon_spd)] = 0.
        latlon_spd[np.isinf(latlon_spd)] = 0.
        return latlon_spd

    def load_track_data(self):
        track_data = pd.read_csv(self.actual_track_datapath, header = 0, index_col = 0)
        # FID, Elap_Time, Lat, Lon, Alt, DT, Speed (nmi/sec), Elap_Time_Diff (sec), course (rad)

        # calculate lat long speed
        latlon_spd = self._calc_latlon_spd(track_data)
        track_data.loc[:, 'Lat_spd'] = latlon_spd[:, 0]
        track_data.loc[:, 'Lon_spd'] = latlon_spd[:, 1]

        # merge with flight plans
        FP_track = pd.read_csv(self.flight_plan_datapath)
        FP_utlize = pd.read_csv(self.flight_plan_utilize_datapath, header = 0)

        # subtract departure airport's [lat, lon] from flight plan (FP) track and standardize
        FP_track[['LATITUDE', 'LONGITUDE']] -= np.array([self.dep_lat, self.dep_lon])
        avg_FP = FP_track[['LATITUDE', 'LONGITUDE']].mean().values
        std_err_FP = FP_track[['LATITUDE', 'LONGITUDE']].std().values
        FP_track[['LATITUDE', 'LONGITUDE']] = (FP_track[['LATITUDE', 'LONGITUDE']] - avg_FP)/std_err_FP

        # merge track data with FP utilize data
        track_data_with_FP_id = track_data.merge(FP_utlize, left_on = 'FID', right_on = 'FID', how = 'inner')
        # process FP tracks
        # Long format to wide format
        FP_track_wide = FP_track.groupby('FLT_PLAN_ID').apply(lambda x: x[['LATITUDE', 'LONGITUDE']].values.reshape(1, -1)).reset_index()
        FP_track_wide.columns = ['FLT_PLAN_ID', 'FP_tracks']
        FP_track_wide['seq_len'] = FP_track_wide.FP_tracks.apply(lambda x: x.shape[1]//2)

        # merge track data with wide form of FP tracks
        track_data_with_FP = track_data_with_FP_id.merge(FP_track_wide, left_on='FLT_PLAN_ID', right_on = 'FLT_PLAN_ID')
        seq_length_tracks = track_data_with_FP.groupby('FID').FLT_PLAN_ID.count().values.astype(np.int32)
        track_data_with_FP['cumDT'] = track_data_with_FP.groupby('FID').DT.transform(pd.Series.cumsum)
        tracks = track_data_with_FP[['Lat', 'Lon', 'Alt', 'cumDT', 'Lat_spd', 'Lon_spd']].values.astype(np.float32)
        print('use cumDT')
        # print('Use absolute time elapsed from: ', baseline_time)
        
        # use delta lat and delta lon
        tracks = tracks - np.array([self.dep_lat, self.dep_lon, 0., 0., 0., 0.])
        avg = tracks.mean(axis = 0)
        std_err = tracks.std(axis = 0)
        tracks = (tracks - avg)/std_err
        tracks_split = np.split(tracks, np.cumsum(seq_length_tracks))[:-1]

        # add the arrival information to construct the target sequence
        targets_split, targets_end_split, targets_end_split_neg = self._construct_target(tracks_split, avg, std_err, self.time_dim)

        FP_track_order = track_data_with_FP.groupby('FID')[['FID', 'Elap_Time', 'FP_tracks', 'seq_len']].head(1)
        seq_length_FP = FP_track_order.seq_len.values.astype(np.int32)
        FP_tracks_split = FP_track_order.FP_tracks.values

        FP_track_order['Elap_Time'] = pd.to_datetime(FP_track_order['Elap_Time'], errors = 'coerce')
        tracks_time_id_info = FP_track_order[['FID', 'Elap_Time']].values

        FP_tracks_split = _pad_and_flip_FP(FP_tracks_split, seq_length_FP)

        # all standardized
        return tracks_split, targets_split, targets_end_split, targets_end_split_neg, seq_length_tracks, avg, std_err, FP_tracks_split, seq_length_FP, avg_FP, std_err_FP, tracks_time_id_info

    def _construct_target(self, splitted_tracks, avg, std_err, time_dim = False):
        tmp_list = []
        tmp_end_list = []
        tmp_end_list_neg = []
        for target_seq in splitted_tracks:
            # print(target_seq.shape)
            # print(avg.shape)
            # print(std_err.shape)
            if time_dim:
                tmp_list.append(np.concatenate((target_seq[1:, :], (np.array([[self.arr_lat - self.dep_lat, 
                                                                              self.arr_lon - self.dep_lon, 
                                                                              0,
                                                                              0, 
                                                                              0, 
                                                                              0]]) - avg)/std_err), axis = 0))
            else:
                tmp_list.append(np.concatenate((target_seq[1:, [0,1,2,4,5]], (np.array([[self.arr_lat - self.dep_lat, 
                                                                                          self.arr_lon - self.dep_lon, 
                                                                                          0,
                                                                                          0, 
                                                                                          0]]) - avg[[0,1,2,4,5]])/std_err[[0,1,2,4,5]]), axis = 0))
            tmp_arr = np.zeros((target_seq.shape[0], 1))
            tmp_arr[-1, 0] = 1.
            tmp_end_list.append(tmp_arr)
            tmp_end_list_neg.append(1 - tmp_arr)
        return tmp_list, tmp_end_list, tmp_end_list_neg


    def load_feature_cubes(self):

        feature_cubes_pointer = np.load(self.feature_cubes_datapath)
        # feature_grid = feature_cubes_pointer['feature_grid']
        # query_idx = feature_cubes_pointer['query_idx']
        feature_cubes = feature_cubes_pointer['feature_cubes']
        # feature_cubes_grid = feature_cubes_pointer['feature_grid'] - np.array([self.dep_lon, self.dep_lat])
        # feature_cubes_grid = feature_cubes_grid.reshape(-1, 20, 20, 2)
        # feature_cubes = np.concatenate((feature_cubes, feature_cubes_grid), axis = -1)
        # feature_cubes have shape of [N_points, 20, 20, 4]
        
        # Standardize the features
        feature_cubes_mean = np.mean(feature_cubes, axis = 0)
        feature_cubes_std = np.std(feature_cubes, axis = 0)
        # Do NOT standardize the binary layer!
        feature_cubes_mean[:, :, 0] = 0.
        feature_cubes_std[:, :, 0] = 1.
        feature_cubes_norm = (feature_cubes - feature_cubes_mean)/feature_cubes_std # shape of [N_point, 20, 20, n_channels]

        return feature_cubes_norm, feature_cubes_mean, feature_cubes_std

    def next_batch(self):
        # n_sample = self.n_train_data_set
        train_dev_test = 'train'
        idx_list = np.arange(self.n_train_data_set)
        if self.idx >= self.n_train_data_set:
            self.idx = 0
            if self.shuffle_or_not:
                idx_list = shuffle(idx_list)

        if train_dev_test == 'train':
            endidx = min(self.idx + self.batch_size, self.n_train_data_set)
            batch_seq_lens = self.train_seq_lens[idx_list[self.idx:endidx]]
            batch_inputs = self.train_tracks[idx_list[self.idx:endidx], :, :]
            batch_targets = self.train_targets[idx_list[self.idx:endidx], :, :]
            batch_targets_end = self.train_targets_end[idx_list[self.idx:endidx], :, :]
            batch_targets_end_neg = self.train_targets_end_neg[idx_list[self.idx:endidx], :, :]

            batch_seq_lens_FP = self.train_seq_lens_FP[idx_list[self.idx:endidx]]
            batch_inputs_FP = self.train_FP_tracks[idx_list[self.idx:endidx], :, :]

            batch_inputs_feature_cubes = self.train_feature_cubes[self.idx:endidx, :, :, :, :]

            self.idx += self.batch_size
            
        return batch_inputs, batch_targets, batch_targets_end, batch_targets_end_neg, batch_seq_lens, batch_inputs_FP, batch_seq_lens_FP, batch_inputs_feature_cubes

#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################

from utils_features import match_wind_fname, match_ncwf_fname, flight_track_feature_generator, proxilvl

class DatasetSample(flight_track_feature_generator):
        
    def __init__(self,
                 train_track_mean,
                 train_track_std,
                 train_fp_mean,
                 train_fp_std,
                 feature_cubes_mean, 
                 feature_cubes_std,
                 ncwf_data_rootdir = '../../DATA/NCWF/gridded_storm_hourly/',
                 test_track_dir = '../../DATA/DeepTP/test_flight_tracks.csv',
                 test_fp_dir = '../../DATA/DeepTP/test_flight_plans.csv',
                 flight_plan_util_dir = '../../DATA/DeepTP/test_flight_plans_util.CSV',
                 wind_data_rootdir = '../../DATA/filtered_weather_data/namanl_small_npz/',
                 grbs_common_info_dir = '/media/storage/DATA/filtered_weather_data/grbs_common_info.npz',
                 grbs_lvl_dict_dir = '/media/storage/DATA/filtered_weather_data/grbs_level_common_info.pkl',
                 grbs_smallgrid_kdtree_dir = '/media/storage/DATA/filtered_weather_data/grbs_smallgrid_kdtree.pkl',
                 ncwf_arr_dir = '../../DATA/NCWF/gridded_storm.npz',
                 ncwf_alt_dict_dir = '../../DATA/NCWF/alt_dict.pkl',
                 large_load = False,
                 weather_feature = True,
                 **kwargs):
        self.train_track_mean = train_track_mean
        self.train_track_std = train_track_std
        self.train_fp_mean = train_fp_mean
        self.train_fp_std = train_fp_std
        self.train_feature_cubes_mean = feature_cubes_mean
        self.train_feature_cubes_std = feature_cubes_std
        self.ncwf_data_rootdir = ncwf_data_rootdir
        self.large_load = large_load
        self.weather_feature = weather_feature

        self.dep_lat = kwargs.get('dep_lat', 29.98333333)
        self.dep_lon = kwargs.get('dep_lon', -95.33333333)
        self.arr_lat = kwargs.get('arr_lat', 42.3666666667)
        self.arr_lon = kwargs.get('arr_lon', -70.9666666667)

        self.direct_course = kwargs.get('direct_course', g.inv(self.dep_lon, self.dep_lat, self.arr_lon, self.arr_lat)[0] * np.pi/180)

        super().__init__(flight_track_dir = test_track_dir,
                         flight_plan_dir = test_fp_dir,
                         flight_plan_util_dir = flight_plan_util_dir,
                         wind_data_rootdir = wind_data_rootdir,
                         grbs_common_info_dir = grbs_common_info_dir,
                         grbs_lvl_dict_dir = grbs_lvl_dict_dir,
                         grbs_smallgrid_kdtree_dir = grbs_smallgrid_kdtree_dir,
                         ncwf_arr_dir = ncwf_arr_dir,
                         ncwf_alt_dict_dir = ncwf_alt_dict_dir,
                         load_ncwf_arr = False,
                         downsample = False)
    def __str__(self):
        return 'Dataset Class to Conduct Sampling Procedure'

    def _calc_latlon_spd(self, track_dataframe):
        CenterTraj = track_dataframe[['FID', 'Lat', 'Lon', 'DT']]
        # CenterTraj.loc[:, 'azimuth'] = last_pnt
        tmp_df = CenterTraj.shift(-1)
        latlon_spd = np.divide((tmp_df[['Lat', 'Lon']].values - CenterTraj[['Lat', 'Lon']].values), tmp_df.DT.values.reshape(-1,1))
        tmp_tail_idx = CenterTraj.groupby("FID")['Lat'].tail(1).index
        latlon_spd[tmp_tail_idx, :] = 0.
        latlon_spd[np.isnan(latlon_spd)] = 0.
        latlon_spd[np.isinf(latlon_spd)] = 0.
        return latlon_spd

    def _count_unordered_seq_length(self, count_array):
        fp_seq_length = []
        _tmp_ = []
        j = -1
        for i in count_array:
            if i not in _tmp_:
                _tmp_.append(i)
                fp_seq_length.append(1)
                j += 1
            else:
                fp_seq_length[j] += 1
        return np.array(fp_seq_length).astype(np.int32)

    def process_test_tracks(self):
        flight_tracks = self.flight_track_preprocess(self.ori_flight_tracks)
        flight_tracks['cumDT'] = flight_tracks.groupby('FID').DT.transform(pd.Series.cumsum)

        # calculate lat long speed
        latlon_spd = self._calc_latlon_spd(flight_tracks)
        flight_tracks.loc[:, 'Lat_spd'] = latlon_spd[:, 0]
        flight_tracks.loc[:, 'Lon_spd'] = latlon_spd[:, 1]
        flight_tracks = flight_tracks.groupby('FID').head(20).reset_index(drop = True)

        # multiple tracks must have the same length for now
        tracks = flight_tracks[['Lat', 'Lon', 'Alt', 'cumDT', 'Lat_spd', 'Lon_spd']].values.astype(np.float32)
        # print('Using Elap_Time_Diff as time inputs')
        # subtract depature's lat lon & course
        # normalize tracks using train mean and train std
        tracks = self.normalize_flight_tracks(tracks)

        # seq_length = flight_tracks.groupby('FID').Lat.count().values.astype(np.int32)
        seq_length = self._count_unordered_seq_length(flight_tracks.FID.values)
        tracks_split = np.split(tracks, np.cumsum(seq_length))[:-1]
        tracks_split = np.array(tracks_split)

        # flight plans
        fp_tracks = self.ori_flight_plans[['LATITUDE', 'LONGITUDE']].values.astype(np.float32)

        # first substract from the lat lon of departure airport
        # then normalize using the training set mean and std
        fp_tracks = (fp_tracks - np.array([self.dep_lat, self.dep_lon]) - self.train_fp_mean)/self.train_fp_std
        fp_seq_length = self._count_unordered_seq_length(self.ori_flight_plans.FLT_PLAN_ID.values)
        # pad and flip
        fp_tracks_split = _pad_and_flip_FP(np.array(np.split(fp_tracks, np.cumsum(fp_seq_length))[:-1]), fp_seq_length)


        return fp_tracks_split, tracks_split, fp_seq_length, seq_length, flight_tracks

    def normalize_flight_tracks(self, 
                                unnormalized_tracks):
        return (unnormalized_tracks - np.array([self.dep_lat, self.dep_lon, 0, 0, 0, 0]) - self.train_track_mean)/self.train_track_std
    def unnormalize_flight_tracks(self,
                                  normalized_tracks):
        return normalized_tracks * self.train_track_std + self.train_track_mean + np.array([self.dep_lat, self.dep_lon, 0, 0, 0, 0])

    def unnormalize_flight_track_cov(self,
                                     normalize_flight_track_cov):
        return normalize_flight_track_cov * (self.train_track_std[[0, 1, 2, 4, 5]] ** 2)

    def normalize_feature_cubes(self,
                                unnormalized_feature_cubes):
        return (unnormalized_feature_cubes - self.train_feature_cubes_mean)/self.train_feature_cubes_std


    # @Override parent method _generate_feature_cube
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
        feature_cubes = np.zeros(shape = (feature_grid_query_idx.shape[0], nx, ny, 4), dtype = np.float32)

        #######################################################################################################
        self.wx_testdata_holder = []
        self.wx_gpidx_holder = []
        if not self.weather_feature:
            feature_cubes[:, :, :, 0] = 0
        else:
            # append all weather data into one so that later matching will be more efficient
            groups = flight_tracks[['FID', 'wx_fname', 'wx_alt']].groupby(['wx_fname', 'wx_alt'])
            ng = groups.ngroups
            print('Extract ncwf convective weather from %d groups ...'%ng)
            for gpidx, gp in groups:
                if gpidx[0] not in self.wx_gpidx_holder:
                    self.wx_gpidx_holder.append(gpidx[0])
                    wx_data_single = self._load_ncwf_low_memory(gpidx[0])
                    self.wx_testdata_holder.append(wx_data_single) # each element is the ncwf array with the order of wx_fname
                else:
                    wx_data_single = self.wx_testdata_holder[self.wx_gpidx_holder.index(gpidx[0])]
                
                # nan has been automatically dropped
                wx_alt_cover = self.wx_unique_alt[(self.wx_unique_alt >= (gpidx[1] - wx_alt_buffer)) & \
                                                  (self.wx_unique_alt <= (gpidx[1] + wx_alt_buffer))]
                wx_alt_idxmin = self.wx_alt_dict[wx_alt_cover.min()]
                wx_alt_idxmax = self.wx_alt_dict[wx_alt_cover.max()] + 1
                wx_base = np.any(wx_data_single[wx_alt_idxmin: wx_alt_idxmax, :][:, feature_grid_query_idx[gp.index]], axis = 0).astype(np.float32).reshape(-1, nx, ny)
                feature_cubes[gp.index, :, :, 0] = wx_base
        print('Finished ncwf wx extraction!\n')

        #######################################################################################################
        groups = flight_tracks[['FID', 'wind_fname', 'levels']].groupby(['wind_fname', 'levels'])
        ng = groups.ngroups
        self.uwind_testdata_holder = []
        self.vwind_testdata_holder = []
        self.tempr_testdata_holder = []

        self.wind_gpidx_holder = []
        print('Extract wind/ temperature from %d groups ...'%ng)
        jj = -1
        for gpidx, gp in groups:
            jj += 1
            if self.large_load:
                wind_npz = np.load(os.path.join(self.wind_data_rootdir, gpidx[0]))
                tmp_uwind = wind_npz['uwind']
                tmp_vwind = wind_npz['vwind']
                tmp_tempr = wind_npz['tempr']
            else:
                if gpidx[0] not in self.wind_gpidx_holder:
                    self.wind_gpidx_holder.append(gpidx[0])
                    tmp_uwind, tmp_vwind, tmp_tempr = self._load_wind_low_memory(gpidx[0])
                    self.tempr_testdata_holder.append(tmp_tempr)
                    self.uwind_testdata_holder.append(tmp_uwind)
                    self.vwind_testdata_holder.append(tmp_vwind)
                    
                else:
                    tmp_tempr = self.tempr_testdata_holder[self.wind_gpidx_holder.index(gpidx[0])]
                    tmp_uwind = self.uwind_testdata_holder[self.wind_gpidx_holder.index(gpidx[0])]
                    tmp_vwind = self.vwind_testdata_holder[self.wind_gpidx_holder.index(gpidx[0])]
                    

            tempr_base = tmp_tempr[self.lvls_dict[gpidx[1]]][feature_grid_query_idx[gp.index]].reshape(-1, nx,ny)
            uwind_base = tmp_uwind[self.lvls_dict[gpidx[1]]][feature_grid_query_idx[gp.index]].reshape(-1, nx,ny)
            vwind_base = tmp_vwind[self.lvls_dict[gpidx[1]]][feature_grid_query_idx[gp.index]].reshape(-1, nx,ny)
            

            feature_cubes[gp.index, :, :, 1] = tempr_base
            feature_cubes[gp.index, :, :, 2] = uwind_base
            feature_cubes[gp.index, :, :, 3] = vwind_base
            
        print('Finished wind/ temperature extraction!\n')
        
        return feature_cubes


    def _load_ncwf_low_memory(self, ncwf_fname):
        return np.load(os.path.join(self.ncwf_data_rootdir, ncwf_fname))['ncwf_arr']

    def generate_test_track_feature_cubes(self,
                                          flight_tracks,
                                          shift_xleft = 0,
                                          shift_xright = 2,
                                          shift_yup = 1,
                                          shift_ydown = 1,
                                          nx = 20,
                                          ny = 20):
        feature_cubes, feature_grid, query_idx = self.feature_arr_generator(flight_tracks = flight_tracks,
                                                                            shift_xleft = shift_xleft,
                                                                            shift_xright = shift_xright,
                                                                            shift_yup = shift_yup,
                                                                            shift_ydown = shift_ydown,
                                                                            nx  = nx,
                                                                            ny = ny)
        # feature_grid = feature_grid - np.array([self.dep_lon, self.dep_lat])
        # feature_grid = feature_grid.reshape(-1, 20, 20, 2)
        # feature_cubes = np.concatenate((feature_cubes, feature_grid), axis = -1)

        feature_cubes = self.normalize_feature_cubes(feature_cubes)

        return feature_cubes, feature_grid, query_idx

    def generate_predicted_pnt_feature_cube(self,
                                            predicted_final_track,
                                            known_flight_deptime,
                                            shift_xleft = 0,
                                            shift_xright = 2,
                                            shift_yup = 1,
                                            shift_ydown = 1,
                                            nx = 20,
                                            ny = 20):
        """
        predicted_final_track has the shape of [n_seq * n_mixture^i, n_time + t, n_input].
            The last axis coresponds to [Lat, Lon, Alt, cumDT, Speed, course]
        known_flight_deptime is a np array that contains
            FID, Elap_Time (depature time)
        wind_file_info is a dictionary of file time tree (kdtree) and an array of time objects

        """
        predicted_final_track = self.unnormalize_flight_tracks(predicted_final_track[:, -2:, :])
        # print(predicted_final_track[0, -1, :4])

        azimuth_arr = g.inv(predicted_final_track[:, -2, 1],
                            predicted_final_track[:, -2, 0],
                            predicted_final_track[:, -1, 1],
                            predicted_final_track[:, -1, 0])[0]
        # Step 0: construct tmp matching dataframe that contains:
        #    elap_time_diff, azimuth, levels, wx_alt, wind_fname, wx_fname
        predicted_matched_info = np.empty((predicted_final_track.shape[0], 13))
        predicted_matched_info = pd.DataFrame(predicted_matched_info, 
                                              columns = ['FID',
                                                         'Lat', 
                                                         'Lon', 
                                                         'Alt', 
                                                         'cumDT', 
                                                         'Lat_spd', 
                                                         'Lon_spd',
                                                         'Elap_Time_Diff', 
                                                         'azimuth', 
                                                         'levels', 
                                                         'wx_alt', 
                                                         'wind_fname', 
                                                         'wx_fname'])

        predicted_matched_info.loc[:, ['Lat', 
                                       'Lon', 
                                       'Alt', 
                                       'cumDT', 
                                       'Lat_spd', 
                                       'Lon_spd']] = predicted_final_track[:, -1, :]

        predicted_matched_info.loc[:, 'azimuth'] = azimuth_arr * np.pi/180
        
        # Step 1: map cumDT to Elaps_time
        known_flight_deptime_diff = (known_flight_deptime[:, 1] - baseline_time)
        known_flight_deptime_diff = np.array([item.total_seconds() for item in known_flight_deptime_diff])
        multiplier = predicted_matched_info.shape[0]//known_flight_deptime_diff.shape[0]
        deptime = np.repeat(known_flight_deptime_diff, repeats = multiplier, axis = 0)
        FIDs =  np.repeat(known_flight_deptime[:, 0], repeats = multiplier, axis = 0)

        elap_time_diff = predicted_matched_info.loc[:, 'cumDT'].values + deptime
        predicted_matched_info.loc[:, 'Elap_Time_Diff'] = elap_time_diff
        predicted_matched_info.loc[:, 'FID'] = FIDs
        
        # Step 2: Map Elaps_time with wx_fname and wind_fname
        # match with wind/ temperature fname
        wind_query_dist, wind_query_idx = self.wind_ftime_tree.query(elap_time_diff.reshape(-1, 1), p = 1, distance_upper_bound = 3600*3)
        wind_valid_query = wind_query_dist < self.wind_time_objs.shape[0] # binary array
        predicted_matched_info.loc[wind_valid_query, 'wind_fname'] = self.wind_time_objs[wind_query_idx[wind_valid_query], 0]
        predicted_matched_info.loc[~wind_valid_query, 'wind_fname'] = np.nan

        # match with ncwf idx
        wx_query_dist, wx_query_idx = self.wx_ftime_tree.query(elap_time_diff.reshape(-1, 1), p = 1, distance_upper_bound = 3600)
        wx_valid_query = wx_query_dist < self.wx_fname_hourly.shape[0] # binary array
        predicted_matched_info.loc[wx_valid_query, 'wx_fname'] = self.wx_fname_hourly[wx_query_idx[wx_valid_query]]
        predicted_matched_info.loc[~wx_valid_query, 'wx_fname'] = np.nan

        # Step 3: calculate wind_levels & ncwf_levels
        predicted_matched_info.loc[:, 'levels'] = predicted_matched_info['Alt'].apply(lambda x: proxilvl(x*100, self.lvls_dict))
        predicted_matched_info.loc[:, 'wx_alt'] = predicted_matched_info['Alt']//10

        # Step 4: generate feature cube
        feature_cubes, feature_grid, _ = self.feature_arr_generator(flight_tracks = predicted_matched_info,
                                                                    shift_xleft = shift_xleft,
                                                                    shift_xright = shift_xright,
                                                                    shift_yup = shift_yup,
                                                                    shift_ydown = shift_ydown,
                                                                    nx  = nx,
                                                                    ny = ny)
        # feature_grid = feature_grid - np.array([self.dep_lon, self.dep_lat])
        # feature_grid = feature_grid.reshape(-1, 20, 20, 2)
        # feature_cubes = np.concatenate((feature_cubes, feature_grid), axis = -1)

        feature_cubes = self.normalize_feature_cubes(feature_cubes)
        feature_cubes = feature_cubes.reshape(-1, 1, nx, ny, 4)
        
        return feature_cubes, feature_grid, predicted_matched_info

    def reshape_feature_cubes(self, 
                              feature_cubes,
                              track_length):
        # track_length should be a list of integers that contains the length of each test track

        feature_cubes = np.array(np.split(feature_cubes, np.cumsum(track_length))[:-1])

        return feature_cubes

def _pad(inputs, inputs_len):
    # inputs is a list of np arrays
    # inputs_len is a np array
    _zero_placeholder = ((0,0),) * (len(inputs[0].shape) - 1)
    max_len = inputs_len.max()
    _inputs = []
    i = 0
    for _input in inputs:
        _tmp_zeros = ((0, max_len - inputs_len[i]), *_zero_placeholder)
        _inputs.append(np.pad(_input, _tmp_zeros, 'constant', constant_values = 0))
        i+=1
    return np.asarray(_inputs)

def _pad_and_flip_FP(inputs, inputs_len):
    # reverse every flight plan
    max_len = inputs_len.max()
    _inputs = []
    i = 0
    for _input in inputs:
        _inputs.append(np.pad(_input.reshape(-1,2)[::-1], ((0, max_len - inputs_len[i]), (0,0)), 'constant', constant_values = 0))
        i+=1
    return np.asarray(_inputs)
