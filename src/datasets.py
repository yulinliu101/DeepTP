import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os
try:
    import cPickle as pickle
except:
    import pickle


#TODO:
# CHANGE COURSE TO COURSE - DIRECT_ROUTE_COURSE

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
        self.actual_track_datapath = actual_track_datapath
        self.flight_plan_datapath = flight_plan_datapath
        self.flight_plan_utilize_datapath = flight_plan_utilize_datapath
        self.feature_cubes_datapath = feature_cubes_datapath
        self.shuffle_or_not = shuffle_or_not
        self.split = split
        self.batch_size = batch_size

        self.dep_lat = kwargs.get('dep_lat', 29.98333333)
        self.dep_lon = kwargs.get('dep_lon', -95.33333333)
        self.direct_course = kwargs.get('direct_course', 0) # TODO
        self.idx = kwargs.get('idx', 0)

        self.all_tracks, \
            self.all_seq_lens, \
                self.data_mean, \
                    self.data_std, \
                        self.all_FP_tracks, \
                            self.all_seq_lens_FP, \
                                self.FP_mean, \
                                    self.FP_std = self.load_track_data()

        self.feature_cubes, self.feature_cubes_mean, self.feature_cubes_std = self.load_feature_cubes()
        self.feature_cubes = np.split(self.feature_cubes, np.cumsum(self.all_seq_lens))[:-1]

        if self.shuffle_or_not:
            self.all_tracks, \
              self.all_seq_lens, \
                self.all_FP_tracks, \
                  self.all_seq_lens_FP, \
                    self.feature_cubes = shuffle(self.all_tracks, 
                                                 self.all_seq_lens, 
                                                 self.all_FP_tracks, 
                                                 self.all_seq_lens_FP, 
                                                 self.feature_cubes,
                                                 random_state = 101)

        if self.split:
            self.train_tracks, \
             self.dev_tracks, \
              self.train_seq_lens, \
               self.dev_seq_lens, \
                self.train_FP_tracks, \
                 self.dev_FP_tracks, \
                  self.train_seq_lens_FP, \
                   self.dev_seq_lens_FP, \
                    self.train_feature_cubes, \
                     self.dev_feature_cubes = train_test_split(self.all_tracks, 
                                                               self.all_seq_lens, 
                                                               self.all_FP_tracks,
                                                               self.all_seq_lens_FP,
                                                               self.feature_cubes,
                                                               random_state = 101, 
                                                               train_size = 0.8,
                                                               test_size = None)

        self.train_tracks = _pad(self.train_tracks, self.train_seq_lens)
        self.train_feature_cubes = _pad(self.train_feature_cubes, self.train_seq_lens)
        self.n_train_data_set = self.train_tracks.shape[0]
    def __str__(self):
        return 'Dataset Class to Conduct Training Procedure'

    def load_track_data(self):
        track_data = pd.read_csv(self.actual_track_datapath, header = 0, usecols = [1, 8, 9, 10, 13, 15, 19], index_col = 0)
        # FID, Lat, Lon, Alt, DT, Speed (nmi/sec), course
        FP_track = pd.read_csv(self.flight_plan_datapath)
        FP_utlize = pd.read_csv(self.flight_plan_utilize_datapath, header = 0, usecols = [19,1])

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
        tracks = track_data_with_FP[['Lat', 'Lon', 'Alt', 'cumDT', 'Speed', 'course']].values.astype(np.float32)
        
        # use delta lat and delta lon
        tracks = tracks - np.array([self.dep_lat, self.dep_lon, 0, 0, 0, 0])
        avg = tracks.mean(axis = 0)
        std_err = tracks.std(axis = 0)
        tracks = (tracks - avg)/std_err
        tracks_split = np.split(tracks, np.cumsum(seq_length_tracks))[:-1]

        FP_track_order = track_data_with_FP.groupby('FID')[['FP_tracks', 'seq_len']].head(1)
        seq_length_FP = FP_track_order.seq_len.values.astype(np.int32)
        FP_tracks_split = FP_track_order.FP_tracks.values

        FP_tracks_split = _pad_and_flip_FP(FP_tracks_split, seq_length_FP)

        # all standardized
        return tracks_split, seq_length_tracks, avg, std_err, FP_tracks_split, seq_length_FP, avg_FP, std_err_FP

    def load_feature_cubes(self):
        # return np.ones((150000, 4, 20, 20)), np.ones((4, 20, 20)), np.ones((4, 20,20))
        feature_cubes_pointer = np.load(self.feature_cubes_datapath)
        # feature_grid = feature_cubes_pointer['feature_grid']
        # query_idx = feature_cubes_pointer['query_idx']
        feature_cubes = feature_cubes_pointer['feature_cubes']
        # feature_cubes = np.transpose(feature_cubes, axes = [0, 3, 1, 2]) 
        # feature_cubes have shape of [N_points, 20, 20, 4]
        
        # Standardize the features
        feature_cubes_mean = np.mean(feature_cubes, axis = 0)
        feature_cubes_std = np.std(feature_cubes, axis = 0)
        # Do NOT standardize the binary layer!
        feature_cubes_mean[0, :, :] = 0.
        feature_cubes_std[0, :, :] = 1.
        feature_cubes_norm = (feature_cubes - feature_cubes_mean)/feature_cubes_std # shape of [N_point, 20, 20, 4]

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

            batch_seq_lens_FP = self.train_seq_lens_FP[idx_list[self.idx:endidx]]
            batch_inputs_FP = self.train_FP_tracks[idx_list[self.idx:endidx], :, :]

            batch_inputs_feature_cubes = self.train_feature_cubes[self.idx:endidx, :, :, :, :]

            self.idx += self.batch_size
            
        batch_targets = None
        return batch_inputs, batch_targets, batch_seq_lens, batch_inputs_FP, batch_seq_lens_FP, batch_inputs_feature_cubes
            

def _pad(inputs, inputs_len):
    # inputs is a list of np arrays
    # inputs_len is a np array
    _zero_placeholder = ((0,0),) * (len(inputs[0].shape)-1)
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



from utils_features import match_wind_fname, match_ncwf_fname, flight_track_feature_generator
class DatasetSample(flight_track_feature_generator):
    def __init__(self,
                 train_track_mean,
                 train_track_std,
                 train_fp_mean,
                 train_fp_std,
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
                 **kwargs):
        self.train_track_mean = train_track_mean
        self.train_track_std = train_track_std
        self.train_fp_mean = train_fp_mean
        self.train_fp_std = train_fp_std
        self.ncwf_data_rootdir = ncwf_data_rootdir

        self.dep_lat = kwargs.get('dep_lat', 29.98333333)
        self.dep_lon = kwargs.get('dep_lon', -95.33333333)
        self.direct_course = kwargs.get('direct_course', 0)

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

    def process_test_tracks(self):
        flight_tracks = self.flight_track_preprocess(self.ori_flight_tracks)
        flight_tracks['cumDT'] = flight_tracks.groupby('FID').DT.transform(pd.Series.cumsum)

        # multiple tracks must have the same length for now
        tracks = flight_tracks[['Lat', 'Lon', 'Alt', 'cumDT', 'Speed', 'course']].values.astype(np.float32)

        # subtract depature's lat lon & course
        # normalize tracks using train mean and train std
        tracks = (tracks - np.array([self.dep_lat, self.dep_lon, 0, 0, 0, self.direct_course]) - self.train_track_mean)/self.train_track_std

        seq_length = flight_tracks.groupby('FID').Lat.count().values.astype(np.int32)
        tracks_split = np.split(tracks, np.cumsum(seq_length))[:-1]
        tracks_split = np.array(tracks_split)

        # flight plans
        fp_tracks = self.ori_flight_plans[['LATITUDE', 'LONGITUDE']].values.astype(np.float32)

        # first substract from the lat lon of departure airport
        # then normalize using the training set mean and std
        fp_tracks = (fp_tracks - np.array([self.dep_lat, self.dep_lon]) - self.train_fp_mean)/self.train_fp_std
        fp_seq_length = self.ori_flight_plans.groupby('FLT_PLAN_ID').LATITUDE.count().values.astype(np.int32)
        # pad and flip
        fp_tracks_split = _pad_and_flip_FP(np.array(np.split(fp_tracks, np.cumsum(fp_seq_length))[:-1]), fp_seq_length)


        return fp_tracks_split, tracks_split, fp_seq_length, seq_length, flight_tracks


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
        # append all weather data into one so that later matching will be more efficient
        groups = flight_tracks[['FID', 'wx_fname', 'wx_alt']].groupby(['wx_fname', 'wx_alt'])
        ng = groups.ngroups

        print('Extract ncwf convective weather from %d groups ...'%ng)
        for gpidx, gp in groups:
            wx_data_single = self._load_ncwf_low_memory(gpidx[0])
            self.wx_testdata_holder.append(wx_data_single) # each element is the ncwf array with the order of wx_fname
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
        print('Extract wind/ temperature from %d groups ...'%ng)
        jj = -1
        for gpidx, gp in groups:
            jj += 1
            # wind_npz = np.load(os.path.join(self.wind_data_rootdir, gpidx[0]))
            # tmp_uwind = wind_npz['uwind']
            # tmp_vwind = wind_npz['vwind']
            # tmp_tempr = wind_npz['tempr']
            tmp_uwind, tmp_vwind, tmp_tempr = self._load_wind_low_memory(gpidx[0])

            self.uwind_testdata_holder.append(tmp_uwind)
            self.vwind_testdata_holder.append(tmp_vwind)
            self.tempr_testdata_holder.append(tmp_tempr)

            uwind_base = tmp_uwind[self.lvls_dict[gpidx[1]]][feature_grid_query_idx[gp.index]].reshape(-1, nx,ny)
            vwind_base = tmp_vwind[self.lvls_dict[gpidx[1]]][feature_grid_query_idx[gp.index]].reshape(-1, nx,ny)
            tempr_base = tmp_tempr[self.lvls_dict[gpidx[1]]][feature_grid_query_idx[gp.index]].reshape(-1, nx,ny)

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


        return feature_cubes, feature_grid, query_idx

    def generate_predicted_pnt_feature_cube(self, 
                                            predicted_point,
                                            known_flight_tracks,
                                            shift_xleft = 0,
                                            shift_xright = 2,
                                            shift_yup = 1,
                                            shift_ydown = 1,
                                            nx = 20,
                                            ny = 20):
        """
        predicted_point has the form of [Lat, Lon, Alt, cumDT, Speed, course], and the shape should be: [?]
        """


        # Step 1: map cumDT to Elaps_time

        # Step 2: calculate azimuth


        # Step 3: Map Elaps_time with wx_fname and wind_fname


        # Step 4: generate feature cube
        TODO

        return










# class Dataset:
#     def __init__(self,
#                  data_path,
#                  shuffle_or_not = True,
#                  split = True,
#                  batch_size = 128,
#                  **kwargs):
#         self.data_path = data_path
#         self.shuffle_or_not = shuffle_or_not
#         self.split = split
#         self.batch_size = batch_size
#         self.idx = kwargs.get('idx', 0)

#         self.all_tracks, self.all_seq_lens, self.data_mean, self.data_std = self.load_data()

#         if self.shuffle_or_not:
#             self.all_tracks, self.all_seq_lens = shuffle(self.all_tracks, self.all_seq_lens, random_state = 101)
#         if self.split:
#             self.train_tracks, self.dev_tracks, self.train_seq_lens, self.dev_seq_lens = train_test_split(self.all_tracks, 
#                                                                                                           self.all_seq_lens, 
#                                                                                                           random_state = 101, 
#                                                                                                           train_size = 0.8)

#         self.train_tracks = _pad(self.train_tracks, self.train_seq_lens)
#         self.n_train_data_set = self.train_tracks.shape[0]
        

#     def load_data(self):
#         track_data = pd.read_csv(self.data_path, header = 0, usecols = [0, 7, 8, 9, 12])
#         seq_length = track_data.groupby('FID').Lat.count().values.astype(np.int32)
#         tracks = track_data[['Lat', 'Lon', 'Alt', 'DT']].values.astype(np.float32)
#         avg = tracks.mean(axis = 0)
#         std_err = tracks.std(axis = 0)
#         tracks = (tracks - avg)/std_err
#         tracks_split = np.split(tracks, np.cumsum(seq_length))[:-1]

#         return tracks_split, seq_length, avg, std_err

#     def next_batch(self):
#         n_sample = self.n_train_data_set
#         train_dev_test = 'train'
#         if self.idx >= n_sample:
#             self.idx = 0
#             if self.shuffle_or_not:
#                 self.train_tracks, self.train_seq_lens = shuffle(self.train_tracks, self.train_seq_lens)
#         if train_dev_test == 'train':
#             endidx = min(self.idx + self.batch_size, self.n_train_data_set)
#             batch_seq_lens = self.train_seq_lens[self.idx:endidx]
#             batch_inputs = self.train_tracks[self.idx:endidx, :, :]
            
#         batch_targets = None
#         return batch_inputs, batch_targets, batch_seq_lens
















# class dataset:
#     def __init__(data_path, 
#                  model = 'rnn',
#                  data_source = 'pickle',
#                  shuffle = True,
#                  **kwargs):
#         self.data_path = data_path
#         self.data_source = data_source
#         self.shuffle = shuffle
#         self.idx = kwargs.get('idx', 0)

#         if self.data_source == 'pickle':
#             self.train_inputs = self.load_data_from_pickle()
#         elif self.data_source == 'zip':
#             self.train_inputs = self.load_data_from_pickle()

#     def load_data_from_zip(self):
            
#         with zipfile(data_path, 'r') as data_file:
#             for fname in data_file.namelist():
#                 img_data = pickle.load(data_file.read(fname))

#     def load_data_from_pickle(self):
#         try:
#             all_img_data = pickle.load(open(data_path, 'rb'))
#         except:
#             all_img_data = pickle.load(open(data_path, 'rb'), protocal = 2, encoding = 'latin1')

#         if self.model == 'autoencoder':
#             return all_img_data, None
#         elif self.model == 'convnet':
#             try:
#                 all_label = pickle.load(open(data_path, 'rb'))
#             except:
#                 all_label = pickle.load(open(data_path, 'rb'), protocal = 2, encoding = 'latin1')
#             return all_img_data, all_label

#     def next_batch(self,
#                    batch_size):
#         if self.idx >= n_sample:
#             self.idx = 0
#             if self.shuffle:
#                 self.train_inputs, self.train_targets = shuffle(self.train_inputs, self.train_targets)

#         if train_dev_test == 'train':
#             endidx = min(self.idx + self.batch_size, self.n_train_data_set)
#             batch_inputs = self.train_inputs[self.idx:endidx, :, :]
#             batch_targets = self.train_targets[self.idx:endidx, :, :]
#             batch_seq_lens = self.train_seq_lens_lens[self.idx:endidx]
#         elif train_dev_test == 'dev':
#             endidx = min(self.idx + self.batch_size, self.n_dev_data_set)
#             batch_inputs = self.dev_inputs[self.idx:endidx, :, :]
#             batch_targets = self.dev_targets[self.idx:endidx, :, :]
#             batch_seq_lens = self.dev_seq_lens_lens[self.idx:endidx]
#         elif train_dev_test == 'test':
#             endidx = min(self.idx + self.batch_size, self.n_prediction_data_set)
#             batch_inputs = self.prediction_inputs[self.idx:endidx, :, :]
#             batch_targets = None
#             batch_seq_lens = self.prediction_seq_lengths[self.idx:endidx]
#         else:
#             raise ValueError('train_dev_test must be train or dev or test')
        
#         self.idx += self.batch_size

#         return batch_inputs, batch_targets, batch_seq_lens