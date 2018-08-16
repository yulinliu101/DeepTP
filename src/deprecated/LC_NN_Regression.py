# -*- coding: utf-8 -*-
"""
Created on Mon May  1 14:26:32 2017

@author: Yulin Liu
"""

try:
    import cPickle as pickle
except:
    import _pickle as pickle
import Fast_dataset_regression
import numpy as np
import tensorflow as tf
import os
import csv

import logging as log
import argparse

IMAGE_DIR = os.getcwd() + '/IAH_BOS_IMG_DATA'
CKPT_DIR = os.getcwd() + '/tmp/tf_logs/ConvNet'
MODEL_CKPT = os.getcwd() + '/tmp/tf_logs/ConvNet/model.cktp'
Point_order = os.getcwd() + '/Point_Order.p'
Pred_Dir = os.getcwd() + '/tmp/PredictLabel'
                      
try:
    os.mkdir(CKPT_DIR)
    os.mkdir(MODEL_CKPT)
    os.mkdir(Pred_Dir)
except:
    pass

# Parameters of Logistic Regression
#BATCH_SIZE = 128

# Network Parameters
IMG_SIZE = 20
n_input = IMG_SIZE**2
n_classes = 2
n_channels = 6
dropout = 0.5 # Dropout, probability to keep units

class ConvNet(object):

    # Constructor
    def __init__(self, learning_rate, max_epochs, display_step, std_dev, X_Feature, Y_Label, BATCH_SIZE):

        # Initialize params
        self.learning_rate=learning_rate
        self.max_epochs=max_epochs
        self.display_step=display_step
        self.std_dev=std_dev
        self.BATCH_SIZE = BATCH_SIZE
        self.X_Feature = X_Feature
        self.Y_Label = Y_Label
        # Store layers weight & bias
        self.weights = {
            'wc1_1': tf.Variable(tf.truncated_normal([9, 9, n_channels, 32], stddev=std_dev)),
            'wc1_2': tf.Variable(tf.truncated_normal([7, 7, 32, 32], stddev=std_dev)),
#            'wc2_1': tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=std_dev)),
#            'wc2_2': tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=std_dev)),
#            'wc3_1': tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=std_dev)),
#            'wc3_2': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=std_dev)),
#            'wc3_3': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=std_dev)),
#            'wc4_1': tf.Variable(tf.random_normal([3, 3, 256, 512], stddev=std_dev)),
#            'wc4_2': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=std_dev)),
#            'wc4_3': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=std_dev)),
#            'wc5_1': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=std_dev)),
#            'wc5_2': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=std_dev)),
#            'wc5_3': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=std_dev)),
            'wd': tf.Variable(tf.truncated_normal([3*3*32, 64])),
            'wfc': tf.Variable(tf.truncated_normal([64, 32], stddev=std_dev)),
            'out': tf.Variable(tf.truncated_normal([32, n_classes], stddev=std_dev))
        }

        self.biases = {
            'bc1_1': tf.Variable(tf.zeros([32])),
            'bc1_2': tf.Variable(tf.zeros([32])),
#            'bc2_1': tf.Variable(tf.zeros([64])),
#            'bc2_2': tf.Variable(tf.zeros([64])),
#            'bc3_1': tf.Variable(tf.zeros([256])),
#            'bc3_2': tf.Variable(tf.zeros([256])),
#            'bc3_3': tf.Variable(tf.zeros([256])),
#            'bc4_1': tf.Variable(tf.zeros([512])),
#            'bc4_2': tf.Variable(tf.zeros([512])),
#            'bc4_3': tf.Variable(tf.zeros([512])),
#            'bc5_1': tf.Variable(tf.zeros([512])),
#            'bc5_2': tf.Variable(tf.zeros([512])),
#            'bc5_3': tf.Variable(tf.zeros([512])),
            'bd': tf.Variable(tf.zeros([64])),
            'bfc': tf.Variable(tf.zeros([32])),
            'out': tf.Variable(tf.zeros([n_classes]))
#            'bc1_1': tf.Variable(tf.random_normal([64])),
#            'bc1_2': tf.Variable(tf.random_normal([64])),
#            'bc2_1': tf.Variable(tf.random_normal([128])),
#            'bc2_2': tf.Variable(tf.random_normal([128])),
#            'bc3_1': tf.Variable(tf.random_normal([256])),
#            'bc3_2': tf.Variable(tf.random_normal([256])),
#            'bc3_3': tf.Variable(tf.random_normal([256])),
#            'bc4_1': tf.Variable(tf.random_normal([512])),
#            'bc4_2': tf.Variable(tf.random_normal([512])),
#            'bc4_3': tf.Variable(tf.random_normal([512])),
#            'bc5_1': tf.Variable(tf.random_normal([512])),
#            'bc5_2': tf.Variable(tf.random_normal([512])),
#            'bc5_3': tf.Variable(tf.random_normal([512])),
#            'bd': tf.Variable(tf.random_normal([4096])),
#            'bfc': tf.Variable(tf.random_normal([4096])),
#            'out': tf.Variable(tf.random_normal([n_classes]))
        }

        # Graph input

        self.img_pl = tf.placeholder(tf.float32, [None, n_input * n_channels])
        self.label_pl = tf.placeholder(tf.float32, [None, n_classes])
        self.keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

        # Create a saver for writing training checkpoints.
        self.saver = tf.train.Saver()
    """
    Create LCNet model
    """
    def conv2d(self, name, l_input, w, b, s):
        # Arbitrary filters that can mix channels together
        # w: fileter = [f_height, f_width, in_channels, out_channels]
        # b: bias term
        # s: strides
        # l_input: input
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, s, s, 1], padding='VALID'), b), name=name)

    def max_pool(self, name, l_input, k, s):
        # l_input: 4d tensor w/ [batch, height, width, channels]
        return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='VALID', name=name)

    def norm(self, name, l_input, lsize=5):
        # 
        return tf.nn.lrn(l_input, depth_radius = lsize, bias=2.0, alpha=0.0001, beta=0.75, name=name)

    def LC_net_model(self, _X, _weights, _biases, _dropout):
        # Similar to Vgg structure, only shallower        
        
        _X = tf.reshape(_X, shape=[-1, IMG_SIZE, IMG_SIZE, n_channels])
        # Convolution Layer 1
        conv1_1 = self.conv2d('conv1_1', _X, _weights['wc1_1'], _biases['bc1_1'], s = 1)
        conv1_2 = self.norm('conv1_2',self.conv2d('conv1_2', conv1_1, _weights['wc1_2'], _biases['bc1_2'], s = 1))
        print("conv1_2.shape: ", conv1_2.get_shape())
        # Max Pooling (down-sampling)
        pool1 = self.max_pool('pool1', conv1_2, k=2, s=2)
        print( "pool1.shape:", pool1.get_shape() )

        # Convolution Layer 2
#        conv2_1 = self.conv2d('conv2_1', pool1, _weights['wc2_1'], _biases['bc2_1'], s = 1)
#        conv2_2 = self.conv2d('conv2_2', conv2_1, _weights['wc2_2'], _biases['bc2_2'], s = 1)
#        print( "conv2_2.shape:", conv2_2.get_shape() )
        # Max Pooling (down-sampling)
#        pool2 = self.max_pool('pool2', conv2_2, k=2, s=2)
#        print( "pool2.shape:", pool2.get_shape() )

#        # Convolution Layer 3
#        conv3_1 = self.conv2d('conv3_1', pool2, _weights['wc3_1'], _biases['bc3_1'], s=1)
#        conv3_2 = self.conv2d('conv3_2', conv3_1, _weights['wc3_2'], _biases['bc3_2'], s=1)
#        conv3_3 = self.conv2d('conv3_3', conv3_2, _weights['wc3_3'], _biases['bc3_3'], s=1)
#        print( "conv3_3.shape:", conv3_3.get_shape())
#        # Max Pooling (down-sampling)
#        pool3 = self.max_pool('pool3', conv3_3, k=2, s=2)
#        print( "pool3.shape:", pool3.get_shape() )
#
#        # Convolution Layer 4
#        conv4_1 = self.conv2d('conv4_1', pool3, _weights['wc4_1'], _biases['bc4_1'], s=1)
#        conv4_2 = self.conv2d('conv4_2', conv4_1, _weights['wc4_2'], _biases['bc4_2'], s=1)
#        conv4_3 = self.conv2d('conv4_3', conv4_2, _weights['wc4_3'], _biases['bc4_3'], s=1)
#        print( "conv4_3.shape:", conv4_3.get_shape())
#        # Max Pooling (down-sampling)
#        pool4 = self.max_pool('pool4', conv4_3, k=2, s=2)
#        print( "pool4.shape:", pool4.get_shape())
#
#        # Convolution Layer 5
#        conv5_1 = self.conv2d('conv5_1', pool4, _weights['wc5_1'], _biases['bc5_1'], s=1)
#        conv5_2 = self.conv2d('conv5_2', conv5_1, _weights['wc5_2'], _biases['bc5_2'], s=1)
#        conv5_3 = self.conv2d('conv5_3', conv5_2, _weights['wc5_3'], _biases['bc4_3'], s=1)
#        print( "conv5_3.shape:", conv5_3.get_shape())
#        # Max Pooling (down-sampling)
#        pool5 = self.max_pool('pool5', conv5_3, k=2, s=2)
#        print( "pool5.shape:", pool5.get_shape() )

        # Fully connected layer 1
        pool1_shape = pool1.get_shape().as_list()
        dense = tf.reshape(pool1, [-1, pool1_shape[1] * pool1_shape[2] * pool1_shape[3]])
        print( "dense.shape:", dense.get_shape())
        fc1 = tf.nn.relu(tf.matmul(dense, _weights['wd']) + _biases['bd'], name='fc1')  # Relu activation
        print( "fc1.shape:", fc1.get_shape() )
        dropout1 = tf.nn.dropout(fc1, _dropout)
        # Fully connected layer 2
        fc2 = tf.nn.relu(tf.matmul(dropout1, _weights['wfc']) + _biases['bfc'], name='fc2')  # Relu activation
        print( "fc2.shape:", fc2.get_shape() )
        dropout2 = tf.nn.dropout(fc2, _dropout)
        # Output, class prediction LOGITS
        y_conv = tf.matmul(dropout2, _weights['out']) + _biases['out']

        softmax_l = tf.nn.softmax(y_conv)

        # The function returns the Logits to be passed to softmax
        return y_conv, softmax_l, fc2



    def training(self):
        X = pickle.load(open(self.X_Feature, 'rb'))
        Y = pickle.load(open(self.Y_Label, 'rb'))
#        config = tf.ConfigProto(
#                device_count = {'GPU': 0}
#            )
#        # Launch the graph
        with tf.Session() as sess:
            # Construct model
            Y_hat, _, _ = self.LC_net_model(self.img_pl, self.weights, self.biases, self.keep_prob)
            cost = tf.reduce_mean(tf.square(Y_hat - self.label_pl))
            
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=0.001).minimize(cost)
            # Evaluate model

            # Initializing the variables
            init = tf.global_variables_initializer()

            # Run the Op to initialize the variables.
            sess.run(init)
            # summary_writer = tf.train.SummaryWriter(CKPT_DIR, graph=sess.graph)
            summary_writer = tf.summary.FileWriter(CKPT_DIR, graph=sess.graph)

            log.info('Dataset created - images list and labels list')
            log.info('Now split images and labels in Training and Test set...')


            ##################################################################
            temp_train_loss = 1e30
            TrainIdx = np.random.choice(range(X.shape[0]), size = int(X.shape[0]*0.95),replace = False)
            TestIdx = np.setdiff1d(range(X.shape[0]), TrainIdx)
            # Run for epoch
            for epoch in range(self.max_epochs):
                
                batchidx = np.random.choice(TrainIdx, size = self.BATCH_SIZE)
                batch = (X[batchidx], Y[batchidx])
                _, train_loss = sess.run([optimizer, cost], feed_dict={self.img_pl: batch[0], 
                                                                       self.label_pl: batch[1], 
                                                                       self.keep_prob: dropout})
                if epoch % self.display_step == 0:
                    train_accuracy = cost.eval(feed_dict={self.img_pl:batch[0], self.label_pl: batch[1], self.keep_prob: 1.0})
                    print("step %d, training accuracy %g"%(epoch, train_accuracy))
            
#                if train_loss > temp_train_loss * 1.25 or epoch % 5000 == 0:
                if epoch % 5000 == 0:
                    self.learning_rate = self.learning_rate * 0.9
                    print( 'Learning Rate Decaying ... %.9f' % self.learning_rate)
                else:
                    temp_train_loss = train_loss
                if epoch % 20 == 0:
                    log.info('Epoch %s' % epoch)
                    log.info("Training Loss = " + "{:.6f}".format(train_loss))

            print( "Optimization Finished!" )

            # Save the models to disk
            save_model_ckpt = self.saver.save(sess, MODEL_CKPT)
            print("Model saved in file %s" % save_model_ckpt) 
            
            # Test accuracy
            # collect imgs for test
            test_acc = cost.eval(feed_dict={self.img_pl: X[TestIdx], 
                                                     self.label_pl: Y[TestIdx], 
                                                     self.keep_prob: 1.0})
            print( "Test accuracy: %.5f" % (test_acc) )
            log.info("Test accuracy: %.5f" % (test_acc))
            
    def prediction(self, NewX):
        with tf.Session() as sess:
            log.info("Start Prediction: %.5f")
#            NewX = self.X[3:6]
#            NewX = NewX.reshape(-1, n_input * n_channels)
            # Construct model
            pred, _, feature_vector = self.LC_net_model(self.img_pl, self.weights, self.biases, self.keep_prob)
    #            prediction = tf.argmax(pred,axis = 1)
    
            # Restore model.
            ckpt = tf.train.get_checkpoint_state(os.getcwd() + "/tmp/tf_logs/ConvNet/")
            if(ckpt):
                self.saver.restore(sess, MODEL_CKPT)
                print("Model restored")
            else:
                print( "No model checkpoint found to restore - ERROR" )
                return
#            for 
            pred_label = sess.run(pred, feed_dict={self.img_pl: NewX, self.keep_prob: 1.})
            extract_feature = sess.run(feature_vector, feed_dict={self.img_pl: NewX, self.keep_prob: 1.})
            
            pickle.dump(pred_label, open(Pred_Dir + '/PredLabel.p', 'wb'), protocol = 2)
            pickle.dump(extract_feature, open(Pred_Dir + '/ExtractFeature.p', 'wb'), protocol = 2)
            
            log.info("File dump to: %s" % Pred_Dir)
        return pred_label, extract_feature

    def prediction_on_train(self):
        Point_idx = pickle.load(open(Point_order,'rb'))
        X = pickle.load(open(self.X_Feature, 'rb'))
        with tf.Session() as sess:
            log.info("Start Prediction: %.5f")
            # Construct model
            pred, _, feature_vector = self.LC_net_model(self.img_pl, self.weights, self.biases, self.keep_prob)
    #            prediction = tf.argmax(pred,axis = 1)
    
            # Restore model.
            ckpt = tf.train.get_checkpoint_state(os.getcwd() + "/tmp/tf_logs/ConvNet/")
            if(ckpt):
                self.saver.restore(sess, MODEL_CKPT)
                print("Model restored")
            else:
                print( "No model checkpoint found to restore - ERROR" )
                return
            
            with open(os.getcwd() + '/PredictedLabelFeature.csv', 'w') as wfile:
                a = csv.writer(wfile)
                for i in range(X.shape[0]):
                    pred_label = sess.run(pred, feed_dict={self.img_pl: X[i].reshape(-1,n_input*n_channels), self.keep_prob: 1.})
                    extract_feature = sess.run(feature_vector, feed_dict={self.img_pl: X[i].reshape(-1,n_input*n_channels), self.keep_prob: 1.})[0]
                    
                    pred_top = list(pred_label.astype(str))
                    extract_feature_w = pred_top + list(extract_feature.astype(str))
                    newline1 = [Point_idx[i][:-2]]
                    newline1.extend(extract_feature_w)
                    a.writerow(newline1)
                    if i % 500 == 0:
                        print(i)
                        print(sum(extract_feature))
#            pickle.dump(pred_label, open(Pred_Dir + '/PredLabel.p', 'wb'), protocol = 2)
#            pickle.dump(extract_feature, open(Pred_Dir + '/ExtractFeature.p', 'wb'), protocol = 2)
            log.info("File written to: %s" % Pred_Dir)
            
### MAIN ###
def main():

    parser = argparse.ArgumentParser(description='A convolutional neural network for heading classification')
    subparsers = parser.add_subparsers()

    common_args = [
        (['-lr', '--learning-rate'], {'help':'learning rate', 'type':float, 'default':0.1}),
        (['-e', '--epochs'], {'help':'epochs', 'type':int, 'default':500}),
        (['-ds', '--display-step'], {'help':'display step', 'type':int, 'default':50}),
        (['-sd', '--std-dev'], {'help':'std-dev', 'type':float, 'default':0.1}),
        (['-d', '--dataset'],  {'help':'dataset file', 'type':str, 'default':'All_img.p'}),
        (['-l', '--labels'],  {'help':'label file', 'type':str, 'default':'All_img_label.p'}),
        (['-BS', '--batch_size'],  {'help':'batch size for optimization', 'type':int, 'default':256})
    ]

    parser_train = subparsers.add_parser('train')
    parser_train.set_defaults(which='train')
    for arg in common_args:
        parser_train.add_argument(*arg[0], **arg[1])

    parser_preprocess = subparsers.add_parser('preprocessing')
    parser_preprocess.set_defaults(which='preprocessing')
    parser_preprocess.add_argument('-f_img', '--IMG_file', help='output file', type=str, default='All_img.p')
    parser_preprocess.add_argument('-f_lab', '--Label_file', help='output 72 label file', type=str, default='All_img_label.p')
#    parser_preprocess.add_argument('-f_lab2', '--Label_file_2', help='output 21 label file', type=str, default='All_img_label_2.p')
#    parser_preprocess.add_argument('-f_lab3', '--Label_file_3', help='output 4 label file', type=str, default='All_img_label_3.p')
    
    
    parser_predict = subparsers.add_parser('predict')
    parser_predict.set_defaults(which='predict')
    parser_predict.add_argument('-NewX', '--NewX_Feature', help='Feed in new data to make prediction')
    
    parser_predict = subparsers.add_parser('predict_on_train')
    parser_predict.set_defaults(which='predict_on_train')
    
    for arg in common_args:
        parser_predict.add_argument(*arg[0], **arg[1])

    args = parser.parse_args()
    if args.which in ('train'):
        log.basicConfig(filename='FileLog.log', level=log.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filemode="w")

    if args.which in ('train', 'predict_on_train'):
        # create the object ConvNet
        conv_net = ConvNet(args.learning_rate, args.epochs, args.display_step, args.std_dev, args.dataset, args.labels, args.batch_size)
        if args.which == 'train':
            # TRAINING
            log.info('Start training')
            conv_net.training()
            print('Prediction on the whole dataset requires huge amount of memory. Do it by chunk!')
        else:
            # PREDICTION
            conv_net.prediction_on_train()
    elif args.which == 'preprocessing':
        Fast_dataset_regression.saveDataset(IMAGE_DIR, args.IMG_file, args.Label_file)

if __name__ == '__main__':
    main()



    
    