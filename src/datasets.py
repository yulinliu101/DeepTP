import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os
try:
    import cPickle as pickle
except:
    import pickle

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
                # self.all_tracks, self.all_seq_lens, self.all_FP_tracks, self.all_seq_lens_FP = shuffle(self.all_tracks, self.all_seq_lens, self.all_FP_tracks, self.all_seq_lens_FP)
                # self.train_tracks, \
                #   self.train_seq_lens, \
                #     self.train_FP_tracks, \
                #       self.train_seq_lens_FP = shuffle(self.train_tracks, 
                #                                        self.train_seq_lens, 
                #                                        self.train_FP_tracks, 
                #                                        self.train_seq_lens_FP)
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