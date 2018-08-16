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

class LSTM_model:
    def __init__(self, conf_path, batch_x, seq_length, n_input, batch_x_decode, seq_length_decode, n_input_decode, target, train = True, weight_summary = False):
        parser = ConfigParser(os.environ)
        parser.read(conf_path)
        n_mixture = parser.getint('lstm', 'n_mixture')
        n_layers = parser.getint('lstm', 'n_lstm_layers')
        self.clipping = parser.getboolean('nn', 'gradient_clipping')
        self.learning_rate = parser.getfloat('nn', 'learning_rate')
        if train:
            dropout = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
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

        n_hidden_1 = parser.getint('lstm', 'n_hidden_1')
        n_hidden_2 = parser.getint('lstm', 'n_hidden_2')
        n_cell_dim = parser.getint('lstm', 'n_cell_dim')
        n_hidden_3 = int(eval(parser.get('lstm', 'n_hidden_3')))
        n_hidden_4 = parser.getint('lstm', 'n_hidden_4')
        
        n_prob_param = parser.getint('lstm', 'n_prob_param')
        n_out = n_prob_param * n_mixture + 1

        # Input shape: [batch_size, n_steps, n_input]
        # # n_input is the # of (original) features per frame: default to be 26
        batch_x_shape = tf.shape(batch_x)
        # # Reshaping `batch_x` to a tensor with shape `[n_steps*batch_size, n_input]`.
        # # This is done to prepare the batch for input into the first layer which expects a tensor of rank `2`.

        batch_x = tf.reshape(batch_x, [-1, n_input])  # (batch_size*time, n_input)
        # clipped RELU activation and dropout.
        # 1st layer
        with tf.name_scope('embedding_encoder'):
            b1 = variable_on_device('b1', [n_hidden_1], tf.random_normal_initializer(stddev=b1_stddev))
            h1 = variable_on_device('h1', [n_input, n_hidden_1],
                                 tf.random_normal_initializer(stddev=h1_stddev))
            layer_emb = tf.nn.elu(tf.nn.xw_plus_b(batch_x, h1, b1))
            # layer_emb = tf.minimum(tf.nn.relu(tf.add(tf.matmul(batch_x, h1), b1)), relu_clip)
            layer_emb = tf.nn.dropout(layer_emb, (1.0 - dropout[0]))

            if train and weight_summary:
                tf.summary.histogram("weights", h1)
                tf.summary.histogram("biases", b1)
                tf.summary.histogram("activations", layer_emb)

        with tf.name_scope('multilayer_encoder'):
            # as the LSTM expects its input to be of shape `[batch_size, time, input_size]`.        
            layer_emb = tf.reshape(layer_emb, [batch_x_shape[0], -1, n_hidden_1])
            # `layer_fc2` is now reshaped into `[n_steps, batch_size, n_hidden_2]`,
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
            

        batch_x_decode_shape = tf.shape(batch_x_decode)
        batch_x_decode = tf.reshape(batch_x_decode, [-1, n_input_decode])  # (batch_size*time, n_input)
        with tf.name_scope('embedding_decoder'):
            b2 = variable_on_device('b2', [n_hidden_2], tf.random_normal_initializer(stddev=b2_stddev))
            h2 = variable_on_device('h2', [n_input_decode, n_hidden_2],
                                 tf.random_normal_initializer(stddev=h2_stddev))
            layer_emb_decode = tf.nn.elu(tf.nn.xw_plus_b(batch_x_decode, h2, b2))
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
                    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = 1 - dropout[2])
                    cells.append(cell)
                stack = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
                if train:
                    self._initial_state = self.encoder_final_state
                else:
                    state_placeholder = tf.placeholder(dtype = tf.float32, shape = [n_layers, 2, None, n_cell_dim], name = 'packed_init_state')
                    unpack_state_placeholder = tf.unstack(state_placeholder, axis=0)
                    self._initial_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(unpack_state_placeholder[idx][0],unpack_state_placeholder[idx][1]) for idx in range(n_layers)])

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
            b_out = variable_on_device('b_out', [n_out], tf.random_normal_initializer(stddev=b_out_stddev))
            h_out = variable_on_device('h_out', [n_cell_dim, n_out], tf.random_normal_initializer(stddev=h_out_stddev))
            layer_out = tf.nn.xw_plus_b(outputs, h_out, b_out) # shape of [batch*time, n_out]            
            # # shape of [batch_size*time, 1 + n_mixture * n_prob_param]
            # n_prob_param = 1 (pi) + 4 (mu) + 10 (cov)
            if train and weight_summary:
                tf.summary.histogram("weights", h_out)
                tf.summary.histogram("biases", b_out)
                tf.summary.histogram("activations", layer_out)

        with tf.name_scope('mixture_coef'):
            self.end_layer, \
             self.pi_layer, \
              self.mu_layer, \
               self.L_layer = get_mixture_coef(layer_out, n_mixture)

        with tf.name_scope('mixture_distribution'):
            self.MVN_pdf = tf.contrib.distributions.MultivariateNormalTriL(loc = self.mu_layer, scale_tril = self.L_layer)

        if not train:
            return

        with tf.name_scope('loss'):
            self.total_loss, self.avg_loss = self.setup_loss(target, n_mixture)
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

    def setup_loss(self, target, n_mixture):
        target_tile = self.reshape_target(target, n_mixture)
        p_i = self.MVN_pdf.prob(target_tile)
        loss = -tf.reduce_sum(tf.log(1e-7+tf.reduce_sum(tf.multiply(self.pi_layer, p_i), axis = 1))) 
        avg_loss = tf.reduce_mean(loss)
        return loss, avg_loss
    
    def reshape_target(self, target, n_mixture):
        # target has shape of [batch,time, 4]
        target_tile = tf.reshape(tf.tile(tf.reshape(target, (-1, 4)), multiples=[1, n_mixture]), (-1, n_mixture, 4))
        return target_tile

def get_mixture_coef(layer_out, n_mixture):
    split_shape = [1, n_mixture, n_mixture * 4, n_mixture * 10]
    end_layer, \
     pi_layer, \
      mu_layer, \
       L_layer = tf.split(value = layer_out, 
                          num_or_size_splits = split_shape,
                          axis = 1)
    
    end_layer = tf.nn.sigmoid(end_layer)
    pi_layer = tf.nn.softmax(pi_layer)

    mu_layer = tf.reshape(mu_layer, (-1, n_mixture, 4)) # [batch*time, n_mixture, 4]
    L_layer = tf.reshape(L_layer, (-1, n_mixture, 10))
    L_layer = tf.contrib.distributions.fill_triangular(L_layer)
    L_layer = tf.contrib.distributions.matrix_diag_transform(L_layer, transform=tf.nn.softplus)
    
    return end_layer, pi_layer, mu_layer, L_layer

# def LSTM(conf_path, batch_x, seq_length, n_input, batch_x_decode, seq_length_decode, n_input_decode, initial_state = None, train = True):
#     # batch_x has the shape of [batch, time, n_dim]

#     parser = ConfigParser(os.environ)
#     parser.read(conf_path)
#     n_mixture = parser.getint('lstm', 'n_mixture')
#     n_layers = parser.getint('lstm', 'n_lstm_layers')
#     if train:
#         dropout = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
#     else:
#         dropout = [0., 0., 0., 0., 0., 0., 0.]

#     b1_stddev = 0.01
#     h1_stddev = 0.01
#     b2_stddev = 0.01
#     h2_stddev = 0.01
#     b3_stddev = 0.01
#     h3_stddev = 0.01
#     b4_stddev = 0.01
#     h4_stddev = 0.01
#     b5_stddev = 0.01
#     h5_stddev = 0.01

#     b_out_stddev = 0.01
#     h_out_stddev = 0.01

#     n_hidden_1 = parser.getint('lstm', 'n_hidden_1')
#     n_hidden_2 = parser.getint('lstm', 'n_hidden_2')
#     n_cell_dim = parser.getint('lstm', 'n_cell_dim')
#     n_hidden_3 = int(eval(parser.get('lstm', 'n_hidden_3')))
#     n_hidden_4 = parser.getint('lstm', 'n_hidden_4')
    
#     n_prob_param = parser.getint('lstm', 'n_prob_param')
#     n_out = n_prob_param * n_mixture + 1
    
#     # Input shape: [batch_size, n_steps, n_input]
#     # # n_input is the # of (original) features per frame: default to be 26
#     batch_x_shape = tf.shape(batch_x)
#     # # Reshaping `batch_x` to a tensor with shape `[n_steps*batch_size, n_input]`.
#     # # This is done to prepare the batch for input into the first layer which expects a tensor of rank `2`.

#     batch_x = tf.reshape(batch_x, [-1, n_input])  # (batch_size*time, n_input)
#     # clipped RELU activation and dropout.
#     # 1st layer
#     with tf.name_scope('embedding_encoder'):
#         b1 = variable_on_device('b1', [n_hidden_1], tf.random_normal_initializer(stddev=b1_stddev))
#         h1 = variable_on_device('h1', [n_input, n_hidden_1],
#                              tf.random_normal_initializer(stddev=h1_stddev))
#         layer_emb = tf.nn.elu(tf.nn.xw_plus_b(batch_x, h1, b1))
#         # layer_emb = tf.minimum(tf.nn.relu(tf.add(tf.matmul(batch_x, h1), b1)), relu_clip)
#         layer_emb = tf.nn.dropout(layer_emb, (1.0 - dropout[0]))

#     with tf.name_scope('multilayer_encoder'):
#         # as the LSTM expects its input to be of shape `[batch_size, time, input_size]`.        
#         layer_emb = tf.reshape(layer_emb, [batch_x_shape[0], -1, n_hidden_1])
#         # `layer_fc2` is now reshaped into `[n_steps, batch_size, n_hidden_2]`,
#         cells_encoder = []
#         with tf.variable_scope('encoder'):
#             for _ in range(n_layers):
#                 cell_encoder = tf.nn.rnn_cell.BasicLSTMCell(n_cell_dim, state_is_tuple = True)
#                 cell_encoder = tf.nn.rnn_cell.DropoutWrapper(cell_encoder, output_keep_prob = 1 - dropout[2])
#                 cells_encoder.append(cell_encoder)
#             stack_encoder = tf.contrib.rnn.MultiRNNCell(cells_encoder, state_is_tuple=True)        
#             # Get layer activations (second output is the final state of the layer, do not need)
#             encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(cell = stack_encoder, 
#                                                                     inputs = layer_emb,
#                                                                     sequence_length = seq_length,
#                                                                     dtype = tf.float32,
#                                                                     time_major=False)
#         # # outputs has the shape of [batch_size, time, n_cell_dim]
#         # # states has the shape of [batch_size, cell.state_size]
#         # # Reshape to apply the same weights over the timesteps
#         # outputs = tf.reshape(outputs, [-1, n_cell_dim]) # [batch*time, n_cell_dim]
        
#     batch_x_decode_shape = tf.shape(batch_x_decode)
#     # # Reshaping `batch_x` to a tensor with shape `[n_steps*batch_size, n_input]`.
#     # # This is done to prepare the batch for input into the first layer which expects a tensor of rank `2`.
#     batch_x_decode = tf.reshape(batch_x_decode, [-1, n_input_decode])  # (batch_size*time, n_input)

#     with tf.name_scope('embedding_decoder'):
#         b2 = variable_on_device('b2', [n_hidden_2], tf.random_normal_initializer(stddev=b2_stddev))
#         h2 = variable_on_device('h2', [n_input_decode, n_hidden_2],
#                              tf.random_normal_initializer(stddev=h2_stddev))
#         layer_emb_decode = tf.nn.elu(tf.nn.xw_plus_b(batch_x_decode, h2, b2))
#         layer_emb_decode = tf.nn.dropout(layer_emb_decode, (1.0 - dropout[0]))

#         # with tf.device('/cpu:0'):
#         tf.summary.histogram("weights", h2)
#         tf.summary.histogram("biases", b2)
#         tf.summary.histogram("activations", layer_emb_decode)

#     with tf.name_scope('multilayer_decoder'):
#         layer_emb_decode = tf.reshape(layer_emb_decode, [batch_x_decode_shape[0], -1, n_hidden_2])
#         cells = []
#         with tf.variable_scope('decoder'):
#             for _ in range(n_layers):
#                 cell = tf.nn.rnn_cell.BasicLSTMCell(n_cell_dim, state_is_tuple = True)
#                 cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = 1 - dropout[2])
#                 cells.append(cell)
#             stack = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
#             if train:
#                 initial_state = encoder_final_state
#             else:
#                 initial_state = initial_state
#             decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(cell = stack, 
#                                                                     inputs = layer_emb_decode,
#                                                                     initial_state = initial_state,
#                                                                     sequence_length = seq_length_decode,
#                                                                     dtype = tf.float32,
#                                                                     time_major=False
#                                                                    )
#         outputs = tf.reshape(decoder_outputs, [-1, n_cell_dim]) # [batch*time, n_cell_dim]
    
#     # with tf.name_scope('fc3'):
#     #     # Now we feed `outputs` to the fifth hidden layer with clipped RELU activation and dropout
#     #     b3 = variable_on_device('b3', [n_hidden_3], tf.random_normal_initializer(stddev=b3_stddev))
#     #     h3 = variable_on_device('h3', [n_cell_dim, n_hidden_3], tf.random_normal_initializer(stddev=h3_stddev))
#     #     layer_fc3 = tf.nn.elu(tf.nn.xw_plus_b(outputs, h3, b3))
#     #     layer_fc3 = tf.nn.dropout(layer_fc3, (1.0 - dropout[4]))
#     #     # with tf.device('/cpu:0'):
#     #     tf.summary.histogram("weights", h3)
#     #     tf.summary.histogram("biases", b3)
#     #     tf.summary.histogram("activations", layer_fc3)

#     # with tf.name_scope('fc4'):
#     #     # Now we feed `outputs` to the fifth hidden layer with clipped RELU activation and dropout
#     #     b4 = variable_on_device('b4', [n_hidden_4], tf.random_normal_initializer(stddev=b4_stddev))
#     #     h4 = variable_on_device('h4', [n_hidden_3, n_hidden_4], tf.random_normal_initializer(stddev=h4_stddev))
#     #     layer_fc4 = tf.nn.elu(tf.nn.xw_plus_b(layer_fc3, h4, b4))
#     #     layer_fc4 = tf.nn.dropout(layer_fc4, (1.0 - dropout[5]))
#     #     # with tf.device('/cpu:0'):
#     #     tf.summary.histogram("weights", h4)
#     #     tf.summary.histogram("biases", b4)
#     #     tf.summary.histogram("activations", layer_fc4)

#     ####################################
#     ##########  LOGIT LAYER  ###########
#     ####################################
#     with tf.name_scope('output_layer'):
#         b_out = variable_on_device('b_out', [n_out], tf.random_normal_initializer(stddev=b_out_stddev))
#         h_out = variable_on_device('h_out', [n_cell_dim, n_out], tf.random_normal_initializer(stddev=h_out_stddev))
#         layer_out = tf.nn.xw_plus_b(outputs, h_out, b_out) # shape of [batch*time, n_out]

        
#         # # shape of [batch_size*time, 1 + n_mixture * n_prob_param]
#         # n_prob_param = 1 (pi) + 4 (mu) + 10 (cov)

#     with tf.name_scope('mixture_coef'):
#         end_layer, \
#          pi_layer, \
#           mu_layer, \
#            L_layer = get_mixture_coef(layer_out, n_mixture)

#     with tf.name_scope('mixture_distribution'):
#         MVN_pdf = tf.contrib.distributions.MultivariateNormalTriL(loc = mu_layer, scale_tril = L_layer)


#     # with tf.device('/cpu:0'):
#     tf.summary.histogram("weights", h1)
#     tf.summary.histogram("biases", b1)
#     tf.summary.histogram("activations", layer_emb)

#     tf.summary.histogram("activations", encoder_outputs)
#     tf.summary.histogram("activations", outputs)

#     tf.summary.histogram("weights", h_out)
#     tf.summary.histogram("biases", b_out)
#     tf.summary.histogram("activations", layer_out)

#     summary_op = tf.summary.merge_all()
#     return end_layer, pi_layer, mu_layer, L_layer, MVN_pdf, encoder_final_state, decoder_final_state, summary_op



    # inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
    # inputs_ta = inputs_ta.unstack(inputs)
    # def loop_fn():
    #     emit_output = cell_output  # == None for time == 0
    #     if cell_output is None:  # time == 0
    #         next_cell_state = initial_state
    #     else:
    #         next_cell_state = cell_state
    #     elements_finished = (time >= seq_length)
    #     finished = tf.reduce_all(elements_finished)
    #     next_input = tf.cond(finished,
    #                          lambda: tf.zeros([batch_size, input_depth], dtype=tf.float32),
    #                          lambda: inputs_ta.read(time))
    #     next_loop_state = None
    #     return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)

    # def train_finished_fn():
    #     return tf.zeros([batch_size, input_depth], dtype=tf.float32)

    # def train_unfinished_fn():
    #     return inputs_ta.read(time)

    # def test_finished_fn():
    #     return tf.zeros([batch_size, input_depth], dtype=tf.float32)

    # def test_unfinished_fn():
    #     return


    # outputs_ta, final_state, _ = raw_rnn(cell, loop_fn)
    # outputs = outputs_ta.stack()
