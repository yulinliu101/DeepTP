import Dataset
import os
import tensorflow as tf
import numpy as np
import logging as log
import argparse
import csv

IMAGE_DIR = os.getcwd() + '/IAH_BOS_IMG_DATA'
#TO_PREDICT_DIR = os.getcwd() + '/images/val'
CKPT_DIR = os.getcwd() + '/tmp/tf_logs/ConvNet'
MODEL_CKPT = os.getcwd() + '/tmp/tf_logs/ConvNet/model.cktp'
try:
    os.mkdir(CKPT_DIR)
    os.mkdir(MODEL_CKPT)
except:
    pass
# Parameters of Logistic Regression
BATCH_SIZE = 128

# Network Parameters
IMG_SIZE = 20
n_input = IMG_SIZE**2
n_classes = 72
n_channels = 6
dropout = 0.5 # Dropout, probability to keep units

class ConvNet(object):

    # Constructor
    def __init__(self, learning_rate, max_epochs, display_step, std_dev, dataset):

        # Initialize params
        self.learning_rate=learning_rate
        self.max_epochs=max_epochs
        self.display_step=display_step
        self.std_dev=std_dev
        self.dataset = dataset
        self.gen_imgs_lab = Dataset.loadDataset(dataset)

        # Store layers weight & bias
        self.weights = {
            'wc1_1': tf.Variable(tf.random_normal([3, 3, n_channels, 64], stddev=std_dev)),
            'wc1_2': tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=std_dev)),
            'wc2_1': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=std_dev)),
            'wc2_2': tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=std_dev)),
#            'wc3_1': tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=std_dev)),
#            'wc3_2': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=std_dev)),
#            'wc3_3': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=std_dev)),
#            'wc4_1': tf.Variable(tf.random_normal([3, 3, 256, 512], stddev=std_dev)),
#            'wc4_2': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=std_dev)),
#            'wc4_3': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=std_dev)),
#            'wc5_1': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=std_dev)),
#            'wc5_2': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=std_dev)),
#            'wc5_3': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=std_dev)),
            'wd': tf.Variable(tf.random_normal([5*5*256, 512])),
            'wfc': tf.Variable(tf.random_normal([512, 256], stddev=std_dev)),
            'out': tf.Variable(tf.random_normal([256, n_classes], stddev=std_dev))
        }

        self.biases = {
            'bc1_1': tf.Variable(tf.zeros([64])),
            'bc1_2': tf.Variable(tf.zeros([128])),
            'bc2_1': tf.Variable(tf.zeros([128])),
            'bc2_2': tf.Variable(tf.zeros([256])),
#            'bc3_1': tf.Variable(tf.zeros([256])),
#            'bc3_2': tf.Variable(tf.zeros([256])),
#            'bc3_3': tf.Variable(tf.zeros([256])),
#            'bc4_1': tf.Variable(tf.zeros([512])),
#            'bc4_2': tf.Variable(tf.zeros([512])),
#            'bc4_3': tf.Variable(tf.zeros([512])),
#            'bc5_1': tf.Variable(tf.zeros([512])),
#            'bc5_2': tf.Variable(tf.zeros([512])),
#            'bc5_3': tf.Variable(tf.zeros([512])),
            'bd': tf.Variable(tf.zeros([512])),
            'bfc': tf.Variable(tf.zeros([256])),
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

        self.img_pl = tf.placeholder(tf.float32, [None, n_input, n_channels])
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
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, s, s, 1], padding='SAME'), b), name=name)

    def max_pool(self, name, l_input, k, s):
        # l_input: 4d tensor w/ [batch, height, width, channels]
        return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='VALID', name=name)

    def norm(self, name, l_input, lsize=5):
        # 
        return tf.nn.lrn(l_input, depth_radius = lsize, bias=2.0, alpha=0.0001, beta=0.75, name=name)

    def LC_net_model(self, _X, _weights, _biases, _dropout):
        # Similar to Vgg structure, only shallower        
        
        _X = tf.reshape(_X, shape=[-1, IMG_SIZE, IMG_SIZE, 6])
        # Convolution Layer 1
        conv1_1 = self.conv2d('conv1_1', _X, _weights['wc1_1'], _biases['bc1_1'], s = 1)
        conv1_2 = self.conv2d('conv1_2', conv1_1, _weights['wc1_2'], _biases['bc1_2'], s = 1)
        print("conv1_2.shape: ", conv1_2.get_shape())
        # Max Pooling (down-sampling)
        pool1 = self.max_pool('pool1', conv1_2, k=2, s=2)
        print( "pool1.shape:", pool1.get_shape() )

        # Convolution Layer 2
        conv2_1 = self.conv2d('conv2_1', pool1, _weights['wc2_1'], _biases['bc2_1'], s = 1)
        conv2_2 = self.conv2d('conv2_2', conv2_1, _weights['wc2_2'], _biases['bc2_2'], s = 1)
        print( "conv2_2.shape:", conv2_2.get_shape() )
        # Max Pooling (down-sampling)
        pool2 = self.max_pool('pool2', conv2_2, k=2, s=2)
        print( "pool2.shape:", pool2.get_shape() )

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
        out = tf.matmul(dropout2, _weights['out']) + _biases['out']

        softmax_l = tf.nn.softmax(out)

        # The function returns the Logits to be passed to softmax
        return out, softmax_l
    """
    Start training
    """       
    # Batch function - give the next batch of images and labels
    def BatchIterator(self, batch_size):
        imgs = []
        labels = []

        for img, label in self.gen_imgs_lab:
            imgs.append(img)
            labels.append(label)
            if len(imgs) == batch_size:
                yield imgs, labels
                imgs = []
                labels = []
        if len(imgs) > 0:
            yield imgs, labels
            
    # Method for training the model and testing its accuracy
    def training(self):
        # Launch the graph
        with tf.Session() as sess:
            # Construct model
            logits, prediction = self.LC_net_model(self.img_pl, self.weights, self.biases, self.keep_prob)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label_pl, logits=logits))
#            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9).minimize(loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=0.001).minimize(loss)

            # Evaluate model
            print( logits.get_shape(), self.label_pl.get_shape() )
            correct_pred = tf.equal(tf.argmax(logits,1), tf.argmax(self.label_pl, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            # Initializing the variables
            init = tf.global_variables_initializer()

            # Run the Op to initialize the variables.
            sess.run(init)
            # summary_writer = tf.train.SummaryWriter(CKPT_DIR, graph=sess.graph)
            summary_writer = tf.summary.FileWriter(CKPT_DIR, graph=sess.graph)

            log.info('Dataset created - images list and labels list')
            log.info('Now split images and labels in Training and Test set...')


            ##################################################################
            # collect imgs for test
            tests_imgs_batches = [b for i, b in enumerate(self.BatchIterator(BATCH_SIZE)) if i < 3]
            temp_train_loss = 1e30
            # Run for epoch
            for epoch in range(self.max_epochs):
                log.info('Epoch %s' % epoch)
                self.gen_imgs_lab = Dataset.loadDataset(self.dataset)

                # Loop over all batches
                for step, elems in enumerate(self.BatchIterator(BATCH_SIZE)):

                    ### from itrator return batch lists ###
                    batch_imgs_train, batch_labels_train = elems
                    # print( batch_imgs_train[0]
                    # print( batch_labels_train[0]
                    # print( batch_labels_train[0].dtype

                    # _, train_acc, train_loss, train_logits = sess.run([optimizer, accuracy, loss, logits], feed_dict={self.img_pl: batch_imgs_train, self.label_pl: batch_labels_train})
                    _, train_acc, train_loss, train_logits = sess.run([optimizer, accuracy, loss, logits], 
                                                                      feed_dict={self.img_pl: batch_imgs_train, 
                                                                                 self.label_pl: batch_labels_train, 
                                                                                 self.keep_prob: dropout})
                    if train_loss > temp_train_loss:
                        self.learning_rate = self.learning_rate * 0.99
                        print( 'Learning Rate Decaying ...')
                    else:
                        temp_train_loss = train_loss
                    
                    log.info("Training Accuracy = " + "{:.5f}".format(train_acc))
                    log.info("Training Loss = " + "{:.6f}".format(train_loss))

            print( "Optimization Finished!" )

            # Save the models to disk
            save_model_ckpt = self.saver.save(sess, MODEL_CKPT)
            print("Model saved in file %s" % save_model_ckpt) 

            # Test accuracy
            for step, elems in enumerate(tests_imgs_batches):
                batch_imgs_test, batch_labels_test = elems

                test_acc = sess.run(accuracy, feed_dict={self.img_pl: batch_imgs_test, self.label_pl: batch_labels_test, self.keep_prob: 1.0})
                print( "Test accuracy: %.5f" % (test_acc) )
                log.info("Test accuracy: %.5f" % (test_acc))


    def prediction(self):
        with tf.Session() as sess:

            # Construct model
            pred, _ = self.LC_net_model(self.img_pl, self.weights, self.biases, self.keep_prob)
#            prediction = tf.argmax(pred,axis = 1)

            # Restore model.
            # ckpt = tf.train.get_checkpoint_state("/tmp/")
            ckpt = tf.train.get_checkpoint_state(os.getcwd() + "/tmp/tf_logs/ConvNet/")
            if(ckpt):
                self.saver.restore(sess, MODEL_CKPT)
                print("Model restored")
            else:
                print( "No model checkpoint found to restore - ERROR" )
                return

            with open(os.getcwd() + '/test.csv', 'w', newline='') as wfile:
                a = csv.writer(wfile)
                for dirName in os.listdir(IMAGE_DIR):
                    path = os.path.join(IMAGE_DIR, dirName)
                    for img in os.listdir(path):
                        print( "reading image to classify... ")
                        img_path = os.path.join(path, img)
                        print("IMG PATH = ", img_path)
                        # check if image is a correct JPG file
                        if(os.path.isfile(img_path) and (img_path.endswith('jpeg') or
                                                         (img_path.endswith('jpg')))):
                            # Read image and convert it
                            img_bytes = tf.read_file(img_path)
                            img_u8 = tf.image.decode_jpeg(img_bytes, channels=3)
                            #img_u8 = tf.image.decode_jpeg(img_bytes, channels=1)
                            img_u8_eval = sess.run(img_u8)
                            image = tf.image.convert_image_dtype(img_u8_eval, tf.float32)
                            img_padded_or_cropped = tf.image.resize_image_with_crop_or_pad(image, IMG_SIZE, IMG_SIZE)
                            img_padded_or_cropped = tf.reshape(img_padded_or_cropped, shape=[IMG_SIZE*IMG_SIZE, 3])
                            #img_padded_or_cropped = tf.reshape(img_padded_or_cropped, shape=[IMG_SIZE * IMG_SIZE])
                            # eval
                            img_eval = img_padded_or_cropped.eval()
                            # Run the model to get predictions
                            predict = sess.run(pred, feed_dict={self.img_pl: [img_eval], self.keep_prob: 1.})
    #                        print( "ConvNet prediction = %s" % (LABELS_DICT.keys()[LABELS_DICT.values().index(predict)])) # Print the name of class predicted
                            pred_top5 = list(predict.argsort(axis = 1)[:,-5:][:,::-1][0].astype(str))
                            newline1 = ['test/'+img] 
                            newline1.extend(pred_top5)
                            a.writerow(newline1)
                        else:
                            print( "ERROR IMAGE:", img_path)

### MAIN ###
def main():

    parser = argparse.ArgumentParser(description='A convolutional neural network for image recognition')
    subparsers = parser.add_subparsers()

    common_args = [
        (['-lr', '--learning-rate'], {'help':'learning rate', 'type':float, 'default':0.05}),
        (['-e', '--epochs'], {'help':'epochs', 'type':int, 'default':2}),
        (['-ds', '--display-step'], {'help':'display step', 'type':int, 'default':10}),
        (['-sd', '--std-dev'], {'help':'std-dev', 'type':float, 'default':0.1}),
        (['-d', '--dataset'],  {'help':'dataset file', 'type':str, 'default':'test_dataset.p'})
    ]

    parser_train = subparsers.add_parser('train')
    parser_train.set_defaults(which='train')
    for arg in common_args:
        parser_train.add_argument(*arg[0], **arg[1])

    parser_preprocess = subparsers.add_parser('preprocessing')
    parser_preprocess.set_defaults(which='preprocessing')
    parser_preprocess.add_argument('-f', '--file', help='output file', type=str, default='images_dataset.p')
    parser_preprocess.add_argument('-s', '--shuffle', help='shuffle dataset', action='store_true')
    parser_preprocess.set_defaults(shuffle=False)

    parser_predict = subparsers.add_parser('predict')
    parser_predict.set_defaults(which='predict')
    for arg in common_args:
        parser_predict.add_argument(*arg[0], **arg[1])

    args = parser.parse_args()
    if args.which in ('train'):
        log.basicConfig(filename='FileLog.log', level=log.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filemode="w")

    if args.which in ('train', 'predict'):
        # create the object ConvNet
        conv_net = ConvNet(args.learning_rate, args.epochs, args.display_step, args.std_dev, args.dataset)
        if args.which == 'train':
            # TRAINING
            log.info('Start training')
            conv_net.training()
        else:
            # PREDICTION
            conv_net.prediction()
    elif args.which == 'preprocessing':
#        if args.shuffle:
#            shuffle(args.file)
#        else:
        Dataset.saveDataset(IMAGE_DIR, args.file)

if __name__ == '__main__':
    main()
