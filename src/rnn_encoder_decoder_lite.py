# Note: All calls to tf.name_scope or tf.summary.* support TensorBoard visualization.

import os
import tensorflow as tf
from configparser import ConfigParser

def variable_on_cpu(name, shape, initializer):
    """
    Next we concern ourselves with graph creation.
    However, before we do so we must introduce a utility function ``variable_on_device()``
    used to create a variable in CPU memory.
    """
    # Use the /cpu:0 device for scoped operations
    with tf.device('/cpu:0'):
        # Create or get apropos variable
        var = tf.get_variable(name=name, shape=shape, initializer=initializer)
    return var

def variable_on_device(name, shape, initializer, device = '/cpu:0'):
    """
    Next we concern ourselves with graph creation.
    However, before we do so we must introduce a utility function ``variable_on_device()``
    used to create a variable in CPU memory.
    """
    # Use the /cpu:0 device for scoped operations
    with tf.device(device):
        var = tf.get_variable(name=name, shape=shape, initializer=initializer)
    return var

def conv2d_bias(inputs, w, b, s, padding = 'SAME'):
    return tf.nn.elu(tf.nn.bias_add(tf.nn.conv2d(inputs, w, strides = [1, s, s, 1], padding = padding), b))

def max_pool(inputs, k, s):
    return tf.nn.max_pool(inputs, ksize = [1, k, k, 1], strides = [1, s, s, 1], padding = 'VALID')

class LSTM_model:
    def __init__(self, 
                 conf_path, 
                 batch_x, 
                 seq_length, 
                 n_input, 
                 batch_x_decode, 
                 batch_xcoords_decode, 
                 seq_length_decode, 
                 n_input_decode, # side size of the image
                 target, 
                 train = True, 
                 weight_summary = False):

        parser = ConfigParser(os.environ)
        parser.read(conf_path)
        n_mixture = parser.getint('lstm', 'n_mixture')
        n_layers = parser.getint('lstm', 'n_lstm_layers')
        n_channels = parser.getint('convolution', 'n_channels')
        self.clipping = parser.getboolean('nn', 'gradient_clipping')
        self.learning_rate = parser.getfloat('nn', 'learning_rate')
        if train:
            dropout = [0.5, 0.5, 0., 0.5, 0.5, 0.5, 0.5]
        else:
            dropout = [0., 0., 0., 0., 0., 0., 0.]

        b1_stddev = 0.01
        h1_stddev = 0.01
        b2_stddev = 0.01
        h2_stddev = 0.01
        b3_stddev = 0.01
        h3_stddev = 0.01
        b4_stddev = 0.01
        h4_stddev = 0.01
        b5_stddev = 0.01
        h5_stddev = 0.01

        b_out_stddev = 0.01
        h_out_stddev = 0.01

        filter_1 = parser.getint('convolution', 'filter_1')
        filter_2 = parser.getint('convolution', 'filter_2')
        filter_3 = parser.getint('convolution', 'filter_3')
        # filter_4 = parser.getint('convolution', 'filter_4')
        filter_1_dep = parser.getint('convolution', 'filter_1_dep')
        filter_2_dep = parser.getint('convolution', 'filter_2_dep')
        filter_3_dep = parser.getint('convolution', 'filter_3_dep')
        # filter_4_dep = parser.getint('convolution', 'filter_4_dep')
        dense_dep = parser.getint('convolution', 'dense_dep')

        wc1_stddev = 0.01
        wc2_stddev = 0.01
        wc3_stddev = 0.01
        # wc4_stddev = 0.01
        wcd_stddev = 0.01
        bc1_stddev = 0.01
        bc2_stddev = 0.01
        bc3_stddev = 0.01
        # bc4_stddev = 0.01
        bcd_stddev = 0.01

        n_hidden_1 = parser.getint('lstm', 'n_hidden_1')
        n_hidden_2 = parser.getint('lstm', 'n_hidden_2')
        n_cell_dim = parser.getint('lstm', 'n_cell_dim')
        # n_hidden_3 = int(eval(parser.get('lstm', 'n_hidden_3')))
        # n_hidden_4 = parser.getint('lstm', 'n_hidden_4')
        n_controled_var = parser.getint('input_dimension', 'n_controled_var')
        n_coords_var = parser.getint('input_dimension', 'n_coords_var')

        n_prob_param = 1 + n_controled_var + sum(range(n_controled_var+1)) + n_coords_var + sum(range(n_coords_var+1))# 1 (pi) + number of elements in (mean vector + upper trig of cov)
        n_out = n_prob_param * n_mixture # ... remove p_end layer

        # Input shape: [batch_size, n_steps, n_input]
        # # n_input is the # of (original) features per frame: default to be 26
        batch_x_shape = tf.shape(batch_x)
        # # Reshaping `batch_x` to a tensor with shape `[n_steps*batch_size, n_input]`.
        # # This is done to prepare the batch for input into the first layer which expects a tensor of rank `2`.

        batch_x = tf.reshape(batch_x, [-1, n_input])  # (batch_size*time, n_input)
        # clipped RELU activation and dropout.
        # 1st layer
        with tf.name_scope('embedding_encoder'):
            b1 = variable_on_device('b1', 
                                    [n_hidden_1], 
                                    tf.random_normal_initializer(stddev=b1_stddev))
            h1 = variable_on_device('h1', 
                                    [n_input, n_hidden_1],
                                    tf.random_normal_initializer(stddev=h1_stddev))
            layer_emb = tf.nn.elu(tf.nn.xw_plus_b(batch_x, h1, b1))
            # layer_emb = tf.minimum(tf.nn.relu(tf.add(tf.matmul(batch_x, h1), b1)), relu_clip)
            # layer_emb = tf.nn.dropout(layer_emb, (1.0 - dropout[0]))

            if train and weight_summary:
                tf.summary.histogram("weights", h1)
                tf.summary.histogram("biases", b1)
                tf.summary.histogram("activations", layer_emb)

        with tf.name_scope('multilayer_encoder'):
            # as the LSTM expects its input to be of shape `[batch_size, time, input_size]`.        
            layer_emb = tf.reshape(layer_emb, [batch_x_shape[0], -1, n_hidden_1])
            cells_encoder = []
            with tf.variable_scope('encoder'):
                for _ in range(n_layers):
                    cell_encoder = tf.nn.rnn_cell.BasicLSTMCell(n_cell_dim, state_is_tuple = True)
                    cell_encoder = tf.nn.rnn_cell.DropoutWrapper(cell_encoder, output_keep_prob = 1 - dropout[2])
                    cells_encoder.append(cell_encoder)
                stack_encoder = tf.contrib.rnn.MultiRNNCell(cells_encoder, state_is_tuple=True)        
                # Get layer activations (second output is the final state of the layer)
                encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(cell = stack_encoder, 
                                                                              inputs = layer_emb,
                                                                              sequence_length = seq_length,
                                                                              dtype = tf.float32,
                                                                              time_major=False)
            # # outputs has the shape of [batch_size, time, n_cell_dim]
            # # states has the shape of [batch_size, cell.state_size]
            # # Reshape to apply the same weights over the timesteps
            # outputs = tf.reshape(outputs, [-1, n_cell_dim]) # [batch*time, n_cell_dim]
            if train and weight_summary:
                tf.summary.histogram("activations", encoder_outputs)

        batch_x_decode_shape = tf.shape(batch_x_decode) #[batch_size, n_time, n_input, n_input, n_channels]
        batch_x_decode = tf.reshape(batch_x_decode, [-1, n_input_decode, n_input_decode, n_channels])  # (batch_size*n_time, n_input, n_input, n_channels)

        with tf.name_scope('embedding_decoder_convolution'):
            wc1 = variable_on_device('wc1', 
                                     [filter_1, filter_1, n_channels, filter_1_dep],
                                     tf.truncated_normal_initializer(stddev=wc1_stddev))
            wc2 = variable_on_device('wc2', 
                                     [filter_2, filter_2, filter_1_dep, filter_2_dep],
                                     tf.truncated_normal_initializer(stddev=wc2_stddev))
            wc3 = variable_on_device('wc3',
                                     [filter_3, filter_3, filter_2_dep, filter_3_dep],
                                     tf.truncated_normal_initializer(stddev=wc3_stddev))
            # wc4 = variable_on_device('wc4',
            #                          [filter_4, filter_4, filter_3_dep, filter_4_dep],
            #                          tf.truncated_normal_initializer(stddev=wc4_stddev))
            final_size = (n_input_decode - filter_1)/2 + 1 - filter_2 + 1 - filter_3 + 1
            wcd = variable_on_device('wcd',
                                     [final_size*final_size*filter_3_dep, dense_dep],
                                     tf.truncated_normal_initializer(stddev=wcd_stddev))

            bc1 = variable_on_device('bc1', 
                                     [filter_1_dep],
                                     tf.random_normal_initializer(stddev=bc1_stddev))
            bc2 = variable_on_device('bc2', 
                                     [filter_2_dep],
                                     tf.random_normal_initializer(stddev=bc2_stddev))
            bc3 = variable_on_device('bc3',
                                     [filter_3_dep],
                                     tf.random_normal_initializer(stddev=bc3_stddev))
            # bc4 = variable_on_device('bc4',
            #                          [filter_4_dep],
            #                          tf.random_normal_initializer(stddev=bc4_stddev))
            bcd = variable_on_device('bcd',
                                     [dense_dep],
                                     tf.random_normal_initializer(stddev=bcd_stddev))

            self.conv1 = conv2d_bias(batch_x_decode, wc1, bc1, s = 2, padding = 'VALID')
            print("conv 1 layer shape: ", self.conv1.get_shape())
            self.conv2 = conv2d_bias(self.conv1, wc2, bc2, s = 1, padding = 'VALID')
            print("conv 2 layer shape: ", self.conv2.get_shape())
            # pool1 = max_pool(conv2, k = 2, s = 2)
            # print("pool 1 layer shape: ", pool1.get_shape())
            self.conv3 = conv2d_bias(self.conv2, wc3, bc3, s = 1, padding = 'VALID')
            print("conv 3 layer shape: ", self.conv3.get_shape())
            # conv4 = conv2d_bias(conv3, wc4, bc4, s = 1, padding = 'VALID')
            # print("conv 4 layer shape: ", conv4.get_shape())
            # pool2 = max_pool(conv4, k = 2, s = 2)
            # print("pool 2 layer shape: ", pool2.get_shape())
            # pool2_shape = tf.shape(pool2)
            conv3_shape = tf.shape(self.conv3)
            self.dense = tf.reshape(self.conv3, [-1, conv3_shape[1]*conv3_shape[2]*conv3_shape[3]])
            print("dense shape (reshape from conv3): ", self.dense.get_shape())
            self.fc1 = tf.nn.elu(tf.nn.xw_plus_b(self.dense, wcd, bcd))
            print("fully connected layer shape: ", self.fc1.get_shape())
            fc1_dropout = tf.nn.dropout(self.fc1, 1 - dropout[1])


        batch_xcoords_decode_shape = tf.shape(batch_xcoords_decode)
        batch_xcoords_decode = tf.reshape(batch_xcoords_decode, [-1, n_controled_var + n_coords_var + 1])  # (batch_size*time, n_coords+n_ctrl+1(time))
        with tf.name_scope('concat_conv_coords'):
            concat_dense = tf.concat([batch_xcoords_decode, fc1_dropout], axis = 1)

        with tf.name_scope('embedding_decoder'):
            b2 = variable_on_device('b2', 
                                    [n_hidden_2], 
                                    tf.random_normal_initializer(stddev=b2_stddev))
            h2 = variable_on_device('h2', 
                                    [dense_dep + n_controled_var + n_coords_var + 1, n_hidden_2],
                                    tf.random_normal_initializer(stddev=h2_stddev))
            layer_emb_decode = tf.nn.elu(tf.nn.xw_plus_b(concat_dense, h2, b2))
            layer_emb_decode = tf.nn.dropout(layer_emb_decode, (1.0 - dropout[0]))

            # with tf.device('/cpu:0'):
            if train and weight_summary:
                tf.summary.histogram("weights", h2)
                tf.summary.histogram("biases", b2)
                tf.summary.histogram("activations", layer_emb_decode)

        with tf.name_scope('multilayer_decoder'):
            layer_emb_decode = tf.reshape(layer_emb_decode, [batch_x_decode_shape[0], -1, n_hidden_2])
            cells = []
            with tf.variable_scope('decoder'):
                for _ in range(n_layers):
                    cell = tf.nn.rnn_cell.BasicLSTMCell(n_cell_dim, state_is_tuple = True)
                    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = 1 - dropout[1])
                    cells.append(cell)
                stack = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

                if train:
                    self._initial_state = self.encoder_final_state
                else:
                    state_placeholder = tf.placeholder(dtype = tf.float32, shape = [n_layers, 2, None, n_cell_dim], name = 'packed_init_state')
                    unpack_state_placeholder = tf.unstack(state_placeholder, axis=0)
                    self._initial_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(unpack_state_placeholder[idx][0],
                                                                               unpack_state_placeholder[idx][1]) for idx in range(n_layers)])

                decoder_outputs, self.decode_final_state = tf.nn.dynamic_rnn(cell = stack, 
                                                                             inputs = layer_emb_decode,
                                                                             initial_state = self._initial_state,
                                                                             sequence_length = seq_length_decode,
                                                                             dtype = tf.float32,
                                                                             time_major=False
                                                                             )
            outputs = tf.reshape(decoder_outputs, [-1, n_cell_dim]) # [batch*time, n_cell_dim]
            if train and weight_summary:
                tf.summary.histogram("activations", outputs)
        ####################################
        ##########  LOGIT LAYER  ###########
        ####################################
        with tf.name_scope('output_layer'):
            b_out = variable_on_device('b_out', 
                                       [n_out], 
                                       tf.random_normal_initializer(stddev=b_out_stddev))
            h_out = variable_on_device('h_out', 
                                       [n_cell_dim, n_out], 
                                       tf.random_normal_initializer(stddev=h_out_stddev))
            layer_out = tf.nn.xw_plus_b(outputs, h_out, b_out) # shape of [batch*time, n_out]            
            # # shape of [batch_size*time, 1 + n_mixture * n_prob_param]
            # n_prob_param = 1 (pi) + n_controled_var (mu) + sum(range(n_controled_var+1)) (cov)
            if train and weight_summary:
                tf.summary.histogram("weights", h_out)
                tf.summary.histogram("biases", b_out)
                tf.summary.histogram("activations", layer_out)

        with tf.name_scope('mixture_coef'):
            
            self.pi_layer, \
              self.mu_layer_coords, \
                self.mu_layer_ctrls, \
                  self.L_layer_coords, \
                    self.L_layer_ctrls = get_mixture_coef(layer_out, n_mixture, n_coord_var = 3, n_controled_var = 2)

        with tf.name_scope('mixture_distribution'):
            self.MVN_pdf_coords = tf.contrib.distributions.MultivariateNormalTriL(loc = self.mu_layer_coords, scale_tril = self.L_layer_coords)
            self.MVN_pdf_ctrls = tf.contrib.distributions.MultivariateNormalTriL(loc = self.mu_layer_ctrls, scale_tril = self.L_layer_ctrls)

        if not train:
            return

        with tf.name_scope('loss'):
            self.total_loss, self.avg_loss = self.setup_loss(target, n_mixture, n_controled_var, n_coords_var)
        # setup optimizer
        with tf.name_scope('training_optimizer'):
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
        if train and weight_summary:
            self.summary_op = tf.summary.merge_all()
        return

    def setup_loss(self, target, n_mixture, n_controled_var, n_coords_var):
        target_coords, target_ctrls = self.reshape_target(target, n_mixture, n_controled_var, n_coords_var)

        p_coords = self.MVN_pdf_coords.prob(target_coords) # shape of [batch * time, n_mixture]
        p_ctrls = self.MVN_pdf_ctrls.prob(target_ctrls) # shape of [batch * time, n_mixture]
        
        p_total = tf.multiply(p_coords, p_ctrls)
        MNV_loss = - tf.log(1e-7 + tf.reduce_sum(tf.multiply(self.pi_layer, p_total), axis = 1, keepdims=True))
        total_loss = tf.reduce_sum(MNV_loss)
        avg_loss = tf.reduce_mean(MNV_loss)

        # self.MNV_loss_coords =  - tf.log(1e-7 + tf.reduce_sum(tf.multiply(self.pi_layer, p_coords), axis = 1, keepdims=True))
        # self.MNV_loss_ctrls =  - tf.log(1e-7 + tf.reduce_sum(tf.multiply(self.pi_layer, p_ctrls), axis = 1, keepdims=True))
        
        # total_loss = tf.reduce_sum(self.MNV_loss_coords + self.MNV_loss_ctrls)
        # avg_loss = tf.reduce_mean(self.MNV_loss_coords + self.MNV_loss_ctrls)
        return total_loss, avg_loss
    
    def reshape_target(self, target, n_mixture, n_controled_var, n_coords_var):
        # target has shape of [batch,time, n_controled_var]
        # target_tile has shape of [batch * time, n_mixture, n_controled_var]
        target_tile = tf.reshape(tf.tile(tf.reshape(target, 
                                                    (-1, n_controled_var + n_coords_var)), 
                                         multiples=[1, n_mixture]), 
                                (-1, n_mixture, n_controled_var + n_coords_var))
        target_coords = tf.gather(target_tile, indices = [0, 1, 2], axis = -1)
        target_ctrls = tf.gather(target_tile, indices = [3, 4], axis = -1)
        return target_coords, target_ctrls

def get_mixture_coef(layer_out, n_mixture, n_coord_var = 3, n_controled_var = 2):
    n_cov_trig_coords = sum(range(n_coord_var + 1))
    n_cov_trig_ctrls = sum(range(n_controled_var + 1))

    split_shape = [n_mixture, n_mixture * n_coord_var, n_mixture * n_controled_var, n_mixture * n_cov_trig_coords, n_mixture * n_cov_trig_ctrls]
    pi_layer, \
      mu_layer_coords, \
       mu_layer_ctrls, \
        L_layer_coords,\
         L_layer_ctrls = tf.split(value = layer_out, 
                                  num_or_size_splits = split_shape,
                                  axis = 1)
    pi_layer = tf.nn.softmax(pi_layer)

    mu_layer_coords = tf.reshape(mu_layer_coords, (-1, n_mixture, n_coord_var)) # [batch*time, n_mixture, n_coord_var]
    L_layer_coords = tf.reshape(L_layer_coords, (-1, n_mixture, n_cov_trig_coords))

    mu_layer_ctrls = tf.reshape(mu_layer_ctrls, (-1, n_mixture, n_controled_var)) # [batch*time, n_mixture, n_controled_var]
    L_layer_ctrls = tf.reshape(L_layer_ctrls, (-1, n_mixture, n_cov_trig_ctrls))

    # Cholesky_decomposition of a P.D. cov matrix is the product of a lower triangular matrix and its conjugate transpose
    L_layer_coords = tf.contrib.distributions.fill_triangular(L_layer_coords)
    L_layer_coords = tf.contrib.distributions.matrix_diag_transform(L_layer_coords, transform=tf.nn.softplus)
    
    L_layer_ctrls = tf.contrib.distributions.fill_triangular(L_layer_ctrls)
    L_layer_ctrls = tf.contrib.distributions.matrix_diag_transform(L_layer_ctrls, transform=tf.nn.softplus)

    return pi_layer, mu_layer_coords, mu_layer_ctrls, L_layer_coords, L_layer_ctrls
