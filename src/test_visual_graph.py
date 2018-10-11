# -*- coding: utf-8 -*-
# @Author: Yulin Liu
# @Date:   2018-10-10 16:53:23
# @Last Modified by:   Yulin Liu
# @Last Modified time: 2018-10-10 17:11:22

import numpy as np
import tensorflow as tf
import os
# import matplotlib.pyplot as plt
from visualize_graph import visual_graph, visualize_raw_weights
import pickle

feature_cubes_pointer = np.load('../data/processed_data/feature_cubes.npz')
feature_cubes = feature_cubes_pointer['feature_cubes']
std_arr_pointer = np.load('../data/processed_data/standardize_arr.npz')
feature_mean = std_arr_pointer['feature_mean']
feature_std = std_arr_pointer['feature_std']
feature_cubes_feed = (feature_cubes - feature_mean)/feature_std



# tf.reset_default_graph()
restored_model_path = 'visual_network/model.ckpt-99'
config_path = 'configs/encoder_decoder_nn.ini'
visual_graph_class = visual_graph(config_path, restored_model_path)

idx = np.random.randint(0, feature_cubes_feed.shape[0], size = 2)
sample_feature_cubes = feature_cubes_feed[idx, None, :, :, :]
print(sample_feature_cubes.shape)

conv1_out, conv2_out, conv3_out = visual_graph_class.feed_fwd_convlayer(feed_input=sample_feature_cubes)
with open('lalala.pkl', 'wb') as pfile:
    pickle.dump(pfile, (conv1_out, conv2_out, conv3_out))

weights = visual_graph_class.weights
visualize_raw_weights(weight_var=weights['wc1:0'], fig_size = (8, 2))
visualize_raw_weights(weight_var=weights['wc2:0'], fig_size = (8,4))
visualize_raw_weights(weight_var=weights['wc3:0'], fig_size = (8,4))
plt.show()