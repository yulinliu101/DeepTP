# -*- coding: utf-8 -*-
# @Author: Yulin Liu
# @Date:   2018-10-10 14:23:23
# @Last Modified by:   Yulin Liu
# @Last Modified time: 2018-10-17 14:11:23

import numpy as np
import tensorflow as tf
import os
from configparser import ConfigParser
from rnn_encoder_decoder_lite import LSTM_model
import matplotlib.pyplot as plt

class visual_graph:
    def __init__(self, 
                 conf_path,
                 restored_model_path):
        self.restored_model_path = restored_model_path
        self.conf_path = conf_path
        self.load_configs()
        
    def load_configs(self):
        parser = ConfigParser(os.environ)
        parser.read(self.conf_path)
        config_header = 'nn'
        self.n_input = parser.getint(config_header, 'n_input')
        self.n_channels = parser.getint('convolution', 'n_channels')
        self.n_controled_var = parser.getint('input_dimension', 'n_controled_var')
        self.n_coords_var = parser.getint('input_dimension', 'n_coords_var')
        self.n_encode = parser.getint(config_header, 'n_encode')
        self.state_size = parser.getint('lstm', 'n_cell_dim')
        self.n_layer = parser.getint('lstm', 'n_lstm_layers')
        # Number of contextual samples to include
        self.batch_size = parser.getint(config_header, 'batch_size')
        
    def define_placeholder(self):
        # define placeholder
        self.input_encode_tensor = tf.placeholder(dtype = tf.float32, shape = [None, None, self.n_encode], name = 'encode_tensor')
        self.seq_len_encode = tf.placeholder(dtype = tf.int32, shape = [None], name = 'seq_length_encode')
        self.input_tensor = tf.placeholder(dtype = tf.float32, shape = [None, None, self.n_input, self.n_input, self.n_channels], name = 'decode_feature_map')
        self.input_decode_coords_tensor = tf.placeholder(dtype = tf.float32, shape = [None, None, self.n_controled_var+self.n_coords_var+1], name = 'decode_coords')
        self.target = tf.placeholder(dtype = tf.float32, shape = [None, None, self.n_controled_var+self.n_coords_var], name = 'target')
        self.seq_length = tf.placeholder(dtype = tf.int32, shape = [None], name = 'seq_length_decode')
        return

    def launchGraph(self):
        self.define_placeholder()
        self.MODEL = LSTM_model(conf_path = self.conf_path,
                                batch_x = self.input_encode_tensor,
                                seq_length = self.seq_len_encode,
                                n_input = self.n_encode,
                                batch_x_decode = self.input_tensor,
                                batch_xcoords_decode = self.input_decode_coords_tensor,
                                seq_length_decode = self.seq_length,
                                n_input_decode = self.n_input,
                                target = self.target,
                                train = False,
                                weight_summary = False)
        
        return
    
    def feed_fwd_convlayer(self, feed_input):
        with tf.device('/cpu:0'):
            self.graph = tf.Graph()
            self.launchGraph()
            self.sess = tf.Session()
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, self.restored_model_path)
            self.sess.graph.finalize()
            self.weights = self._return_weights()
            conv1_out, conv2_out, conv3_out, dense_out = self._feed_fwd_convlayer(feed_input)
            self.sess.close()
        return conv1_out, conv2_out, conv3_out, dense_out
    
    def _return_weights(self):
        weight_list = tf.trainable_variables()
        weights = {}
        for v in weight_list:
            weights[v.name] = self.sess.run(v)
        return weights
    
    def _feed_fwd_convlayer(self, feed_input):
        # feed_input should have the shape of [?, ?, 20, 20, 4]
        conv1_out = self.sess.run(self.MODEL.conv1, feed_dict={self.input_tensor: feed_input})
        conv2_out = self.sess.run(self.MODEL.conv2, feed_dict={self.input_tensor: feed_input})
        conv3_out = self.sess.run(self.MODEL.conv3, feed_dict={self.input_tensor: feed_input})
        dense_out = self.sess.run(self.MODEL.fc1, feed_dict={self.input_tensor: feed_input})
        return conv1_out, conv2_out, conv3_out, dense_out

def visualize_raw_weights(weight_var, fig_size = (8, 4)):
    n_layers = weight_var.shape[3]
    n_channels = weight_var.shape[2]
    fig, axs = plt.subplots(n_channels, n_layers, figsize=fig_size, facecolor='w', edgecolor='k')
    axs = axs.ravel()
    for i in range(n_channels):
        for j in range(n_layers):
            axs[n_layers * i + j].imshow(weight_var[:, :, i, j], 
                                   cmap = 'bwr',
                                   vmax = weight_var.max(), 
                                   vmin = weight_var.min())
            axs[n_layers * i + j].set_axis_off()
    plt.show()
    return fig

def visualize_conv_layers(conv_layer, 
                          nrow, 
                          ncol, 
                          fig_size):
    print(conv_layer.shape)
    # n_layers = weight_var.shape[3]
    # n_channels = weight_var.shape[2]
    fig, axs = plt.subplots(nrow, ncol, figsize=fig_size, facecolor='w', edgecolor='k')
    fig.subplots_adjust(wspace = 0.01, hspace = 0.01)
    axs = axs.ravel()
    for i in range(nrow):
        for j in range(ncol):
            axs[ncol * i + j].imshow(conv_layer[j, :, :, i], 
                                   cmap = 'bwr',
                                   vmax = conv_layer[:, :, :, i].max(), 
                                   vmin = conv_layer[:, :, :, i].min(),
                                   origin = 'lower')
            axs[ncol * i + j].set_axis_off()
    plt.show()
    return fig

'''
Example Code:
'''
'''
tf.reset_default_graph()
restored_model_path = 'visual_network/model.ckpt-99'
config_path = 'configs/encoder_decoder_nn.ini'
visual_graph_class = visual_graph(config_path, restored_model_path)
visual_graph_class.restore_model()
weights = visual_graph_class.weights

visualize_raw_weights(weight_var=weights['wc1:0'], fig_size = (8, 2))
visualize_raw_weights(weight_var=weights['wc2:0'], fig_size = (8,4))
visualize_raw_weights(weight_var=weights['wc3:0'], fig_size = (8,4))
'''