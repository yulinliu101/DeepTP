import tensorflow as tf
import numpy as np
import os
from configparser import ConfigParser
from rnn import LSTM
from datasets import Dataset
import logging
import time
from tensorflow.python.client import device_lib
from sklearn.metrics import confusion_matrix
import pickle
import math


def get_available_gpus():
    """
    Returns the number of GPUs available on this system.
    """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def check_if_gpu_available(gpu_name):
    """
    Returns boolean of if a specific gpu_name (string) is available
    On the system
    """
    list_of_gpus = get_available_gpus()
    if gpu_name not in list_of_gpus:
        return False
    else:
        return True
class trainRNN:
    def __init__(self, 
                 conf_path,
                 sample_traj = False,
                 model_name=None
                 ):

        self.conf_path = conf_path
        self.sample_traj = sample_traj
        self.model_name = model_name
        self.load_configs()
        self.set_up_directories(self.model_name)
        # if don't have gpu, then set device to be cpu
        if not check_if_gpu_available(self.tf_device):
            self.tf_device = '/cpu:0'
        logging.info('Using device %s for main computations', self.tf_device)
        
        self.cpu_dataset = Dataset(data_path = '../data/New_IAHBOS2013.csv',
                                   shuffle = True,
                                   split = True,
                                   batch_size = self.batch_size)

    def load_configs(self):
        parser = ConfigParser(os.environ)
        if not os.path.exists(self.conf_path):
            raise IOError("Configuration file '%s' does not exist" % self.conf_path)
        logging.info('Loading config from %s', self.conf_path)
        parser.read(self.conf_path)

        # set which set of configs to import
        config_header = 'nn'
        logger.info('config header: %s', config_header)
        self.epochs = parser.getint(config_header, 'epochs')
        logger.debug('self.epochs = %d', self.epochs)
        # number of feature length
        self.n_input = parser.getint(config_header, 'n_input')
        self.state_size = parser.getint('lstm', 'n_cell_dim')
        self.n_layer = parser.getint('lstm', 'n_lstm_layers')
        # Number of contextual samples to include
        self.batch_size = parser.getint(config_header, 'batch_size')
        logger.debug('self.batch_size = %d', self.batch_size)
        self.model_dir = parser.get(config_header, 'model_dir')
        self.data_dir = parser.get(config_header, 'data_dir')
        self.n_mixture = parser.getint('lstm', 'n_mixture')
        logger.debug('self.n_mixture = %d', self.n_mixture)
        self.clipping = parser.getboolean(config_header, 'gradient_clipping')
        self.shuffle_data_after_epoch = parser.getboolean(config_header, 'shuffle_data_after_epoch')
        # set the session name
        self.session_name = '{}_{}'.format('LSTM', time.strftime("%Y%m%d-%H%M%S"))
        sess_prefix_str = 'develop'
        if len(sess_prefix_str) > 0:
            self.session_name = '{}_{}'.format(sess_prefix_str, self.session_name)

        # How often to save the model
        self.SAVE_MODEL_EPOCH_NUM = parser.getint(config_header, 'SAVE_MODEL_EPOCH_NUM')
        self.VALIDATION_EPOCH_NUM = parser.getint(config_header, 'VALIDATION_EPOCH_NUM')
        
        # set up GPU if available
        self.tf_device = str(parser.get(config_header, 'tf_device'))
        # optimizer
        self.beta1 = parser.getfloat(config_header, 'beta1')
        self.beta2 = parser.getfloat(config_header, 'beta2')
        self.epsilon = parser.getfloat(config_header, 'epsilon')
        self.learning_rate = parser.getfloat(config_header, 'learning_rate')
        logger.debug('self.learning_rate = %.6f', self.learning_rate)

    def set_up_directories(self, model_name):
        # Set up model directory
        self.model_dir = os.path.join(os.getcwd(), self.model_dir)
        # summary will contain logs
        self.SUMMARY_DIR = os.path.join(
            self.model_dir, "summary", self.session_name)
        # session will contain models
        self.SESSION_DIR = os.path.join(
            self.model_dir, "session", self.session_name)

        if not self.sample_traj:
            if not os.path.exists(self.SESSION_DIR):
                os.makedirs(self.SESSION_DIR)
            if not os.path.exists(self.SUMMARY_DIR):
                os.makedirs(self.SUMMARY_DIR)

        # set the model name and restore if not None
        if model_name is not None:
            tmpSess = os.path.join(self.model_dir, "session")
            self.restored_model_path = os.path.join(tmpSess, model_name)
        else:
            self.restored_model_path = None

    def run_model(self, 
                  train_from_model = False,
                  test_data_start_track = None):
                # define a graph
        self.graph = tf.Graph()
        with self.graph.as_default(), tf.device('/cpu:0'):
            with tf.device(self.tf_device):
                self.launchGraph()
            self.setup_loss_summary()
            self.sess = tf.Session()

            self.writer = tf.summary.FileWriter(self.SUMMARY_DIR, graph=self.sess.graph)

            # Add ops to save and restore all the variables
            self.saver = tf.train.Saver()
            section = '\n{0:=^40}\n'
            if self.restored_model_path is None:
                self.sess.run(tf.global_variables_initializer())
                logger.info("===============================================================")
                # logger.info("Load data into a queue ...")
                # self.sess.run(self.iterator.initializer, feed_dict={self.input_tensor: self.cpu_dataset.train_tracks,
                #                                                     self.seq_length:self.cpu_dataset.train_seq_lens,
                #                                                     self.target:self.cpu_dataset.train_tracks,
                #                                                     self.BATCH_SIZE: self.batch_size})
                self.total_samples = self.cpu_dataset.n_train_data_set
                logger.info("Total training sample size is %d", self.total_samples)
                logger.info("===============================================================")
                logger.info("Start Training ...")
                self.run_training_epoch()
            else:
                self.saver.restore(self.sess, self.restored_model_path)
                if train_from_model is True:
                    logger.info("===============================================================")
                    logger.info(section.format('Run training epoch from restored model %s'%self.restored_model_path))
                    self.total_samples = self.cpu_dataset.n_train_data_set
                    logger.info("Total training sample size is %d", self.total_samples)
                    logger.info("===============================================================")
                    logger.info("Start Training ...")
                    self.run_training_epoch()
                else:
                    logger.info("===============================================================")
                    logger.info(section.format('Restore model from %s'%self.restored_model_path))
                    if self.sample_traj:
                        import pandas as pd
                        track_data = pd.read_csv('../data/test_data.csv', header = 0)
                        tracks = track_data[['Lat', 'Lon', 'Alt', 'DT']].values.astype(np.float32)
                        seq_length = track_data.groupby('SEQ').Lat.count().values.astype(np.int32)
                        tracks_split = np.split(tracks, np.cumsum(seq_length))[:-1]
                        tracks_split = np.array(tracks_split)

                        logger.info("=============== Start sampling ... ==============")
                        predicted_trakcs = self.sample_seq(start_tracks = tracks_split, 
                                                           standard_mu = self.cpu_dataset.data_mean, 
                                                           standard_std = self.cpu_dataset.data_std, 
                                                           max_length = 100, 
                                                           beam_search = False)
                        with open('lalala.pkl', 'wb') as wpkl:
                            pickle.dump(predicted_trakcs, wpkl)
                        
                    else:
                        pass
            # save train summaries to disk
            self.writer.flush()
            self.sess.close()

    def define_placeholder(self):
        # define placeholder
        targetLength = 4
        self.input_tensor = tf.placeholder(dtype = tf.float32, shape = [None, None, self.n_input], name = 'feature_map')
        self.target = tf.placeholder(dtype = tf.float32, shape = [None, None, targetLength], name = 'target')
        self.seq_length = tf.placeholder(dtype = tf.int32, shape = [None], name = 'seq_length')
        self.state_placeholder = tf.placeholder(dtype = tf.float32, shape = [self.n_layer, 2, None, self.state_size], name = 'packed_init_state')
        unpack_state_placeholder = tf.unstack(self.state_placeholder, axis=0)
        with tf.name_scope('initial_state_placeholder'):
            self.initial_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(unpack_state_placeholder[idx][0],unpack_state_placeholder[idx][1]) for idx in range(self.n_layer)])
        # self.BATCH_SIZE = tf.placeholder(dtype = tf.int64, name = 'batch_size')

    def dataset_FIFO_queue(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.input_tensor, self.seq_length, self.target))
        if self.shuffle_data_after_epoch:
            dataset = dataset.shuffle(buffer_size = 10000)
        dataset = dataset.batch(self.BATCH_SIZE)
        dataset = dataset.repeat(self.epochs)
        return dataset
    def launchGraph(self):
        self.define_placeholder()
        # self.dataset = self.dataset_FIFO_queue()
        # self.iterator = self.dataset.make_initializable_iterator()
        # self.next_input_tensor, self.next_seq_length, self.next_target = self.iterator.get_next()

        self.end_layer, \
         self.pi_layer, \
          self.mu_layer, \
           self.L_layer, \
             self.MVN_pdf, \
             self.final_state, \
              summary_op = LSTM(self.conf_path, self.input_tensor, self.seq_length, self.n_input, self.initial_state, not self.sample_traj)

        self.summary_op = tf.summary.merge([summary_op])

        with tf.name_scope('loss'):
            self.total_loss, self.avg_loss = self.setup_loss()

        # setup optimizer
        with tf.name_scope('training_optimizer'):
            # train_ops = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
            #                                    beta1=self.beta1,
            #                                    beta2=self.beta2,
            #                                    epsilon=self.epsilon)
            train_ops = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                   momentum = 0.9,
                                                   use_nesterov = True)
            if self.clipping:
                tvars = tf.trainable_variables()
                self.grads = tf.gradients(self.avg_loss, tvars)
                clipped_grads, _ = tf.clip_by_global_norm(self.grads, 5)
                self.optimizer = train_ops.apply_gradients(zip(clipped_grads, tvars))
            else:
                self.optimizer = train_ops.minimize(self.avg_loss)
        return 

    def setup_loss(self):
        with tf.name_scope('tile_target'):
            target_tile = self.reshape_target(self.target)
        # self.MVN_pdf = tf.contrib.distributions.MultivariateNormalFullCovariance(loc = mu, covariance_matrix = cov)
        # self.MVN_pdf = tf.contrib.distributions.MultivariateNormalTriL(loc = self.mu_layer, scale_tril = self.L_layer)
        p_i = self.MVN_pdf.prob(target_tile)
        # nominator = 1e-9 + tf.exp(-0.5*tf.reshape(tf.matmul(tf.matmul(tf.expand_dims(mu-target_tile, 2), cov_inv), tf.expand_dims(mu-target_tile, 3)), (-1, self.n_mixture)))
        # denominator = 1e-9 + tf.sqrt(((2*math.pi) ** 4)*cov_det)
        # p_i = 1e-7 + tf.divide(nominator, denominator) # size of [batch*time, self.n_mixture]
        loss = -tf.reduce_sum(tf.log(1e-7+tf.reduce_sum(tf.multiply(self.pi_layer, p_i), axis = 1))) 
        avg_loss = tf.reduce_mean(loss)
        return loss, avg_loss

    def setup_loss_summary(self):
        self.loss_placeholder = tf.placeholder(dtype = tf.float32, shape = [])
        self.loss_summary = tf.summary.scalar("training_avg_loss", self.loss_placeholder)   

    def reshape_target(self, target):
        # target has shape of [batch,time, 4]
        target_tile = tf.reshape(tf.tile(tf.reshape(target, (-1, 4)), multiples=[1, self.n_mixture]), (-1, self.n_mixture, 4))
        return target_tile
    
    def run_training_epoch(self):
        train_start_time = time.time()
        for epoch in range(self.epochs):
            is_checkpoint_step, is_validation_step = self.validation_and_checkpoint_check(epoch)
            epoch_start_time = time.time()
            train_epoch_loss = self.run_batches(self.cpu_dataset, 
                                                self.total_samples, 
                                                epoch, 
                                                'train')
            epoch_elap_time = time.time() - epoch_start_time

            log = 'Epoch {}/{}, train_cost: {:.3f}, elapsed_time: {:.2f} sec \n'
            logger.info(log.format(epoch + 1, self.epochs, train_epoch_loss, epoch_elap_time))
            
            summary_line = self.sess.run(self.loss_summary, feed_dict = {self.loss_placeholder: train_epoch_loss})
            self.writer.add_summary(summary_line, epoch)

            # if (epoch + 1 == self.epochs) or is_validation_step:
            #     logger.info('==============================')
            #     logger.info('Validating ...')
            #     dev_accuracy = self.run_dev_epoch(epoch)
            #     logger.info('==============================')

            if (epoch + 1 == self.epochs) or is_checkpoint_step:
                save_path = self.saver.save(self.sess, os.path.join(self.SESSION_DIR, 'model.ckpt'), epoch)
                logger.info("Model saved to {}".format(save_path))
        train_elap_time = time.time() - train_start_time
        logger.info('Training complete, total duration: {:.2f} min'.format(train_elap_time / 60))
        return

    def validation_and_checkpoint_check(self,
                                        epoch):
        # initially set at False unless indicated to change
        is_checkpoint_step = False
        is_validation_step = False
        # Check if the current epoch is a validation or checkpoint step
        if (epoch > 0) and ((epoch + 1) != self.epochs):
            if (epoch + 1) % self.SAVE_MODEL_EPOCH_NUM == 0:
                is_checkpoint_step = True
            if (epoch + 1) % self.VALIDATION_EPOCH_NUM == 0:
                is_validation_step = True

        return is_checkpoint_step, is_validation_step

    def run_batches(self, 
                    dataset,
                    total_samples,
                    epoch,
                    train_dev_test = 'train'
                    ):
        n_batches_per_epoch = total_samples//self.batch_size + 1
        total_training_loss = 0
        for _ in range(n_batches_per_epoch):
            batch_inputs, batch_targets, batch_seq_lens = dataset.next_batch()
            batch_init_state = np.zeros(shape = (self.n_layer, 2, batch_seq_lens.shape[0], self.state_size))
            feeds = {self.input_tensor: batch_inputs,
                     self.target: batch_inputs,
                     self.seq_length: batch_seq_lens,
                     self.state_placeholder: batch_init_state}

            if train_dev_test == 'train':
                # total_batch_loss, _, summary_line = self.sess.run([self.total_loss, self.optimizer, self.summary_op])
                total_batch_loss, _, summary_line = self.sess.run([self.total_loss, self.optimizer, self.summary_op], feed_dict = feeds)
                total_training_loss += total_batch_loss
                logger.debug('Total batch loss: %2.f |Total train cost so far: %.2f', total_batch_loss, total_training_loss)
            self.writer.add_summary(summary_line, epoch)
        return total_training_loss

    def sample_seq(self, start_tracks, standard_mu, standard_std, max_length = 100, beam_search = True):
        # start_tracks should have the shape of [n_sample, n_time, n_input]
        # start_tracks should be standardized with standard_mu and standard_std
        start_tracks = (start_tracks - standard_mu)/standard_std
        n_seq, n_time, _ = start_tracks.shape
        init_state = np.zeros(shape = (self.n_layer, 2, n_seq, self.state_size))
        feeds = {self.input_tensor: start_tracks,
                 self.seq_length: [n_time]*n_seq,
                 self.state_placeholder: init_state}
        state = self.sess.run(self.final_state, feed_dict = feeds)
        last_input_track_point = start_tracks[:, -1, :].reshape(n_seq, 1, -1)

        # pi_placeholder = tf.placeholder(dtype = tf.float32, shape = [None, self.n_mixture])
        pi_MNsample = tf.multinomial(logits = self.pi_layer, num_samples = 1, output_dtype = tf.int32)

        for i in range(max_length - n_time):
            feeds_update = {self.input_tensor: last_input_track_point,
                            self.seq_length: [1]*(n_seq),
                            self.state_placeholder: state}
            state, param_sample, pi_sample = self.sess.run([self.final_state, self.MVN_pdf.sample(), pi_MNsample], feed_dict = feeds_update)
            last_input_track_point = param_sample[range(n_seq), pi_sample.flatten()].reshape(n_seq, 1, -1)
            start_tracks = np.concatenate((start_tracks, last_input_track_point), axis = 1)
        final_tracks = (start_tracks * standard_std) + standard_mu

        return final_tracks


# to run in console
if __name__ == '__main__':
    import click
    # Use click to parse command line arguments
    @click.command()
    @click.option('--train_or_predict', type=str, default='train', help='Train the model or predict model based on input')
    @click.option('--config', default='configs/neural_network.ini', help='Configuration file name')
    @click.option('--name', default=None, help='Path for retored model')
    @click.option('--train_from_model', type=bool, default=False, help='train from restored model')

    # for prediction
    @click.option('--test_data', default='../data/test_data.csv', help='test data path')

    # Train RNN model using a given configuration file
    def main(config='configs/neural_network.ini',
             name = None,
             train_from_model = False,
             train_or_predict = 'train',
             test_data = '../data/test_data.csv'):
        log_name = '{}_{}_{}'.format('log/log', train_or_predict, time.strftime("%Y%m%d-%H%M%S"))
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                            filename=log_name + '.log',
                            filemode='w')
        global logger
        logger = logging.getLogger(os.path.basename(__file__))
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
        logger.addHandler(consoleHandler)

        # create the Tf_train_ctc class
        if train_or_predict == 'train':
            tmpBinary = True
        elif train_or_predict == 'predict':
            tmpBinary = False
        else:
            raise ValueError('train_or_predict not valid')
        tf_train = trainRNN(conf_path=config,
                            model_name=name, 
                            sample_traj = not tmpBinary)
        
        if tmpBinary:
            # run the training
            tf_train.run_model(train_from_model = train_from_model)
        else:
            tf_train.run_model(train_from_model = False,
                               test_data_start_track = test_data)
           
    main()













#     def __construct_normal_stat__(self):
#         # [batch_size * time, self.n_mixture]
#         s11 = tf.square(self.sigma_lat_layer)
#         s22 = tf.square(self.sigma_lon_layer)
#         s33 = tf.square(self.sigma_alt_layer)
#         s44 = tf.square(self.sigma_st_layer)

#         s12 = tf.multiply(tf.multiply(self.sigma_lat_layer, self.sigma_lon_layer), self.rho_latlon_layer)
#         s13 = tf.multiply(tf.multiply(self.sigma_lat_layer, self.sigma_alt_layer), self.rho_altlat_layer)
#         s14 = tf.multiply(tf.multiply(self.sigma_lat_layer, self.sigma_st_layer),  self.rho_st_lat)
#         s23 = tf.multiply(tf.multiply(self.sigma_lon_layer, self.sigma_alt_layer), self.rho_lonalt_layer)
#         s24 = tf.multiply(tf.multiply(self.sigma_lon_layer, self.sigma_st_layer),  self.rho_st_lon)
#         s34 = tf.multiply(tf.multiply(self.sigma_alt_layer, self.sigma_st_layer),  self.rho_st_alt)

#         cov_flat = tf.concat([s11, s12, s13, s14, s12, s22, s23, s24, s13, s23, s33, s34, s14, s24, s34, s44], axis = 1)
#         cov = tf.reshape(tf.transpose(tf.reshape(cov_flat, (-1, 4*4, self.n_mixture)), perm=[0, 2, 1]), (-1, self.n_mixture,4,4))
#         # cov has the shape of [batch*time, self.n_mixture, 4, 4]
#         # cov_det = tf.matrix_determinant(cov) # shape of [batch*time, self.n_mixture]
#         # # self.cov_inv = tf.linalg.inv(self.cov) # shape of [batch*time, self.n_mixture, 4, 4]
#         # const_rhs = tf.eye(num_rows = 4, batch_shape = tf.shape(cov)[:2])
#         # cov_inv = tf.matrix_solve_ls(cov, const_rhs, l2_regularizer = 0.00001)
#         cov_inv = None
#         cov_det = None
        
#         mu_flat = tf.concat([self.mu_lat_layer, self.mu_lon_layer, self.mu_alt_layer, self.mu_st_layer], axis = 1) # [batch*time, self.n_mixture * 4]
#         mu = tf.transpose(tf.reshape(mu_flat, (-1, 4, self.n_mixture)), perm = [0,2,1])
#         # mu has the shape of [batch*time, self.n_mixture, 4]
#         return mu, cov, cov_inv, cov_det