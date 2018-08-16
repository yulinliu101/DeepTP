# -*- coding: utf-8 -*-
"""
Created on Tue May  2 14:59:18 2017

@author: Yulin Liu
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 16:44:25 2017

@author: Yulin Liu
"""
try:
    import cPickle as pickle
except:
    import _pickle as pickle
import Fast_dataset
import numpy as np
import tensorflow as tf
import os
import csv
import logging as log
import argparse
import time

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
n_classes = 4
n_channels = 6
_dropout = 0.5 # Dropout, probability to keep units

# Graph input

img_pl = tf.placeholder(tf.float32, [None, n_input * n_channels])
label_pl = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)


def conv2d(name, l_input, w, b, s):
        # Arbitrary filters that can mix channels together
        # w: fileter = [f_height, f_width, in_channels, out_channels]
        # b: bias term
        # s: strides
        # l_input: input
    return tf.nn.tanh(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, s, s, 1], padding='SAME'), b), name=name)

def max_pool(name, l_input, k, s):
    # l_input: 4d tensor w/ [batch, height, width, channels]
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='VALID', name=name)

def norm(name, l_input, lsize=5):
    # 
    return tf.nn.lrn(l_input, depth_radius = lsize, bias=2.0, alpha=0.0001, beta=0.75, name=name)

def LC_net_model(_X, std_dev):
    # Similar to Vgg structure, only shallower
    global _dropout
    _weights = {
        'wc1_1': tf.Variable(tf.truncated_normal([9, 9, n_channels, 32], stddev=std_dev)),
        'wc1_2': tf.Variable(tf.truncated_normal([9, 9, 32, 64], stddev=std_dev)),
        'wc2_1': tf.Variable(tf.truncated_normal([5, 5, 64, 64], stddev=std_dev)),
        'wc2_2': tf.Variable(tf.truncated_normal([5, 5, 64, 128], stddev=std_dev)),
        'wd': tf.Variable(tf.truncated_normal([IMG_SIZE*IMG_SIZE*128/16, 128])),
        'wfc': tf.Variable(tf.truncated_normal([128, 64], stddev=std_dev)),
        'out': tf.Variable(tf.truncated_normal([64, n_classes], stddev=std_dev))
    }

    _biases = {
        'bc1_1': tf.Variable(tf.zeros([32])),
        'bc1_2': tf.Variable(tf.zeros([64])),
        'bc2_1': tf.Variable(tf.zeros([64])),
        'bc2_2': tf.Variable(tf.zeros([128])),
        'bd': tf.Variable(tf.zeros([128])),
        'bfc': tf.Variable(tf.zeros([64])),
        'out': tf.Variable(tf.zeros([n_classes]))
    }
    _X = tf.reshape(_X, shape=[-1, IMG_SIZE, IMG_SIZE, n_channels])
    # Convolution Layer 1
    conv1_1 = conv2d('conv1_1', _X, _weights['wc1_1'], _biases['bc1_1'], s = 1)
    conv1_2 = conv2d('conv1_2', conv1_1, _weights['wc1_2'], _biases['bc1_2'], s = 1)
    print("conv1_2.shape: ", conv1_2.get_shape())
    # Max Pooling (down-sampling)
    pool1 = max_pool('pool1', conv1_2, k=2, s=2)
    print( "pool1.shape:", pool1.get_shape() )

    # Convolution Layer 2
    conv2_1 = conv2d('conv2_1', pool1, _weights['wc2_1'], _biases['bc2_1'], s = 1)
    conv2_2 = conv2d('conv2_2', conv2_1, _weights['wc2_2'], _biases['bc2_2'], s = 1)
    print( "conv2_2.shape:", conv2_2.get_shape() )
    # Max Pooling (down-sampling)
    pool2 = max_pool('pool2', conv2_2, k=2, s=2)
    print( "pool2.shape:", pool2.get_shape() )
    # Fully connected layer 1
    pool2_shape = pool2.get_shape().as_list()
    dense = tf.reshape(pool2, [-1, pool2_shape[1] * pool2_shape[2] * pool2_shape[3]])
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
    saver = tf.train.Saver()
    
    # The function returns the Logits to be passed to softmax
    return y_conv, softmax_l, fc2, saver



def train_neural_network(X_feature, Y_label, std_dev, learning_rate=0.001, hm_epochs = 1000, BATCH_SIZE = 256):
    x = tf.placeholder('float', [None, n_input * n_channels])
    y = tf.placeholder('float', [None, n_classes])
    
    prediction,_,fc2, saver = LC_net_model(x, std_dev)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
#        temp_train_loss = 1e30
        TrainIdx = np.random.choice(range(X_feature.shape[0]), size = int(X_feature.shape[0]*0.95),replace = False)
        TestIdx = np.setdiff1d(range(X_feature.shape[0]), TrainIdx)
        st = time.time()
        for epoch in range(hm_epochs):
            TrainIdx = np.random.permutation(TrainIdx)
#            batchidx = np.random.choice(TrainIdx, size = BATCH_SIZE, replace = False)
#            batch = (X_feature[batchidx], Y_label[batchidx])
            epoch_loss = 0
            
            for kk in range(int(TrainIdx.shape[0]/BATCH_SIZE)):
                batch = (X_feature[TrainIdx[kk * BATCH_SIZE:(kk+1)*BATCH_SIZE],:], 
                         Y_label[TrainIdx[kk * BATCH_SIZE:(kk+1)*BATCH_SIZE],:])
            

                _, c = sess.run([optimizer, cost], feed_dict={x: batch[0], y: batch[1]})
                epoch_loss += c
                train_acc = accuracy.eval(feed_dict = {x: batch[0], y: batch[1]})
                
                if kk % 20 == 0:
                    print('Epoch', epoch, 'completed out of',hm_epochs,'training acc: ', train_acc)
            if epoch % 10 == 0:
                learning_rate = learning_rate * 0.5
                print('Learning Rate Decaying...: ', learning_rate)
#            if epoch % 10 == 0:
            print('time', time.time() - st, 'Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss, 'training acc: ', train_acc)
            
        save_model_ckpt = saver.save(sess, MODEL_CKPT)
        print("Model saved in file %s" % save_model_ckpt)
        test_acc = accuracy.eval(feed_dict={x: X_feature[TestIdx], y: Y_label[TestIdx]})
        print( "Test accuracy: %.5f" % (test_acc) )        
        
        
        
def Predict_on_train(X_feature, Y_label, PointOrder):
    x = tf.placeholder('float', [None, n_input*n_channels])
    prediction,_,fc2, _ = LC_net_model(x, 0.0001)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        new_saver = tf.train.import_meta_graph(CKPT_DIR + '/model.cktp.meta')
        ckpt = tf.train.get_checkpoint_state(os.getcwd() + "/tmp/tf_logs/ConvNet/")
        if(ckpt):
            new_saver.restore(sess, MODEL_CKPT)
            print("Model restored")
        else:
            print( "No model checkpoint found to restore - ERROR" )
            return
        
#        Idx = np.random.permutation(range(X_feature.shape[0]))
        with open(os.getcwd() + '/PredictedLabelFeature.csv', 'w') as wfile:
            a = csv.writer(wfile)
            for i in range(X_feature.shape[0]):
                pred_label = sess.run(prediction, feed_dict={x: X_feature[i].reshape(-1,n_input*n_channels)})
                extract_feature = sess.run(fc2, feed_dict={x: X_feature[i].reshape(-1,n_input*n_channels)})[0]
                
                pred_top3 = list(pred_label.argsort(axis = 1)[:,-3:][:,::-1][0].astype(str))
                extract_feature_w = pred_top3 + list(extract_feature.astype(str))
                newline1 = [PointOrder[i][:-2],str(np.where(Y_label[i] == 1)[0][0])]
                newline1.extend(extract_feature_w)
                a.writerow(newline1)
                if i % 500 == 0:
                    print(i)
                    print(sum(extract_feature))

def train_from_restore(X_feature, Y_label, std_dev = 0.001, learning_rate=0.0005, hm_epochs = 1000, BATCH_SIZE = 256):
    x = tf.placeholder('float', [None, n_input * n_channels])
    y = tf.placeholder('float', [None, n_classes])
    
    prediction,_,fc2, saver = LC_net_model(x, std_dev)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        new_saver = tf.train.import_meta_graph(CKPT_DIR + '/model.cktp.meta')
        ckpt = tf.train.get_checkpoint_state(os.getcwd() + "/tmp/tf_logs/ConvNet/")
        if(ckpt):
            new_saver.restore(sess, MODEL_CKPT)
            print("Model restored")
        else:
            print( "No model checkpoint found to restore - ERROR" )
            return
        
        TrainIdx = np.random.choice(range(X_feature.shape[0]), size = int(X_feature.shape[0]*0.95),replace = False)
        TestIdx = np.setdiff1d(range(X_feature.shape[0]), TrainIdx)
        st = time.time()
        for epoch in range(hm_epochs):
            TrainIdx = np.random.permutation(TrainIdx)
#            batchidx = np.random.choice(TrainIdx, size = BATCH_SIZE, replace = False)
#            batch = (X_feature[batchidx], Y_label[batchidx])
            epoch_loss = 0
            
            for kk in range(int(TrainIdx.shape[0]/BATCH_SIZE)):
                batch = (X_feature[TrainIdx[kk * BATCH_SIZE:(kk+1)*BATCH_SIZE],:], 
                         Y_label[TrainIdx[kk * BATCH_SIZE:(kk+1)*BATCH_SIZE],:])
            

                _, c = sess.run([optimizer, cost], feed_dict={x: batch[0], y: batch[1]})
                epoch_loss += c
                train_acc = accuracy.eval(feed_dict = {x: batch[0], y: batch[1]})
                
                if kk % 20 == 0:
                    print('Epoch', epoch, 'completed out of',hm_epochs,'training acc: ', train_acc)
            if epoch % 50 == 0:
                learning_rate = learning_rate * 0.75
                print('Learning Rate Decaying...: ', learning_rate)
#            if epoch % 10 == 0:
            print('time', time.time() - st, 'Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss, 'training acc: ', train_acc)
            
        save_model_ckpt = saver.save(sess, MODEL_CKPT)
        print("Model saved in file %s" % save_model_ckpt)
        test_acc = accuracy.eval(feed_dict={x: X_feature[TestIdx], y: Y_label[TestIdx]})
        print( "Test accuracy: %.5f" % (test_acc) )   
    
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#X_feature = mnist.train.images
#Y_label = mnist.train.labels

#X_feature = pickle.load(open('All_img.p', 'rb'))
##X_feature = scale(X_feature,axis = 0)
#X_feature = X_feature/100
#Y_label = pickle.load(open('All_img_label_3.p', 'rb'))
#PointOrder = pickle.load(open('Point_Order.p', 'rb'))
#train_neural_network(X_feature, Y_label, 0.01, 0.005, 300000)
#Predict_on_train(X_feature, Y_label, PointOrder)


def main():

    parser = argparse.ArgumentParser(description='A convolutional neural network for heading classification')
    subparsers = parser.add_subparsers()

    common_args = [
        (['-lr', '--learning_rate'], {'help':'learning rate', 'type':float, 'default':0.001}),
        (['-e', '--epochs'], {'help':'epochs', 'type':int, 'default':500}),
        (['-ds', '--display_step'], {'help':'display step', 'type':int, 'default':10}),
        (['-sd', '--std_dev'], {'help':'std-dev', 'type':float, 'default':0.01}),
        (['-d', '--dataset'],  {'help':'dataset file', 'type':str, 'default':'All_img.p'}),
        (['-l', '--labels'],  {'help':'label file', 'type':str, 'default':'All_img_label_3.p'}),
        (['-BS', '--batch_size'],  {'help':'batch size for optimization', 'type':int, 'default':256})
    ]

    parser_train = subparsers.add_parser('train')
    parser_train.set_defaults(which='train')
    
    parser_train = subparsers.add_parser('train_from_restore')
    parser_train.set_defaults(which='train_from_restore')
    
    for arg in common_args:
        parser_train.add_argument(*arg[0], **arg[1])

    parser_preprocess = subparsers.add_parser('preprocessing')
    parser_preprocess.set_defaults(which='preprocessing')
    parser_preprocess.add_argument('-f_img', '--IMG_file', help='output file', type=str, default='All_img.p')
    parser_preprocess.add_argument('-f_lab', '--Label_file', help='output 72 label file', type=str, default='All_img_label.p')
    parser_preprocess.add_argument('-f_lab2', '--Label_file_2', help='output 21 label file', type=str, default='All_img_label_2.p')
    parser_preprocess.add_argument('-f_lab3', '--Label_file_3', help='output 4 label file', type=str, default='All_img_label_3.p')
    
#    parser_predict = subparsers.add_parser('predict')
#    parser_predict.set_defaults(which='predict')
#    parser_predict.add_argument('-NewX', '--NewX_Feature', help='Feed in new data to make prediction')
    
    parser_predict = subparsers.add_parser('predict_on_train')
    parser_predict.set_defaults(which='predict_on_train')
    
    for arg in common_args:
        parser_predict.add_argument(*arg[0], **arg[1])

    args = parser.parse_args()
    if args.which == 'train':
        X_feature = pickle.load(open(args.dataset, 'rb'))
        X_feature = X_feature/100
        Y_label = pickle.load(open(args.labels, 'rb'))
        train_neural_network(X_feature, Y_label, args.std_dev, args.learning_rate, args.epochs,args.batch_size)
        
    elif args.which == 'predict_on_train':
        X_feature = pickle.load(open(args.dataset, 'rb'))
        X_feature = X_feature/100
        Y_label = pickle.load(open(args.labels, 'rb'))
        PointOrder = pickle.load(open('Point_Order.p', 'rb'))
        Predict_on_train(X_feature, Y_label, PointOrder)
    elif args.which == 'train_from_restore':
        X_feature = pickle.load(open(args.dataset, 'rb'))
        X_feature = X_feature/100
        Y_label = pickle.load(open(args.labels, 'rb'))
        train_from_restore(X_feature, Y_label, args.std_dev, args.learning_rate, args.epochs,args.batch_size)
        
    elif args.which == 'preprocessing':
        Fast_dataset.saveDataset(IMAGE_DIR, args.IMG_file, args.Label_file, args.Label_file_2, args.Label_file_3)

if __name__ == '__main__':
    main()




    
    