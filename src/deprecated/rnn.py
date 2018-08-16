# Note: All calls to tf.name_scope or tf.summary.* support TensorBoard visualization.

import os
import tensorflow as tf
from configparser import ConfigParser
# from keras.layers import CuDNNLSTM

# from models.RNN.utils import variable_on_gpu

def variable_on_cpu(name, shape, initializer):
    """
    Next we concern ourselves with graph creation.
    However, before we do so we must introduce a utility function ``variable_on_gpu()``
    used to create a variable in CPU memory.
    """
    # Use the /cpu:0 device for scoped operations
    with tf.device('/cpu:0'):
        # Create or get apropos variable
        var = tf.get_variable(name=name, shape=shape, initializer=initializer)
    return var

def variable_on_gpu(name, shape, initializer):
    """
    Next we concern ourselves with graph creation.
    However, before we do so we must introduce a utility function ``variable_on_gpu()``
    used to create a variable in CPU memory.
    """
    # Use the /cpu:0 device for scoped operations
    with tf.device('/device:GPU:0'):
        # Create or get apropos variable
        var = tf.get_variable(name=name, shape=shape, initializer=initializer)
    return var

def LSTM(conf_path, batch_x, seq_length, n_input, initial_state = None, train = True):
    # batch_x has the shape of [batch, time, n_dim]

    parser = ConfigParser(os.environ)
    parser.read(conf_path)
    n_mixture = parser.getint('lstm', 'n_mixture')
    n_layers = parser.getint('lstm', 'n_lstm_layers')
    relu_clip = parser.getint('lstm', 'relu_clip')
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
    # # Reshaping `batch_x` to a tensor with shape `[n_steps*batch_size, n_input + 2*n_input*n_context]`.
    # # This is done to prepare the batch for input into the first layer which expects a tensor of rank `2`.

    # # Permute n_steps and batch_size
    # batch_x = tf.transpose(batch_x, [1, 0, 2])
    # Reshape to prepare input for first layer
    # batch_x = tf.reshape(batch_x, [-1, n_input])  # (n_steps*batch_size, n_input)

    batch_x = tf.reshape(batch_x, [-1, n_input])  # (batch_size*time, n_input)
    # clipped RELU activation and dropout.
    # 1st layer
    with tf.name_scope('embedding'):
        b1 = variable_on_gpu('b1', [n_hidden_1], tf.random_normal_initializer(stddev=b1_stddev))
        h1 = variable_on_gpu('h1', [n_input, n_hidden_1],
                             tf.random_normal_initializer(stddev=h1_stddev))
        layer_emb = tf.nn.elu(tf.nn.xw_plus_b(batch_x, h1, b1))
        # layer_emb = tf.minimum(tf.nn.relu(tf.add(tf.matmul(batch_x, h1), b1)), relu_clip)

        layer_emb = tf.nn.dropout(layer_emb, (1.0 - dropout[0]))

        # with tf.device('/cpu:0'):
        tf.summary.histogram("weights", h1)
        tf.summary.histogram("biases", b1)
        tf.summary.histogram("activations", layer_emb)

    with tf.name_scope('fc2'):
        b2 = variable_on_gpu('b2', [n_hidden_2], tf.random_normal_initializer(stddev=b2_stddev))
        h2 = variable_on_gpu('h2', [n_hidden_1, n_hidden_2],
                             tf.random_normal_initializer(stddev=h2_stddev))
        layer_fc2 = tf.nn.elu(tf.nn.xw_plus_b(layer_emb, h2, b2))
        layer_fc2 = tf.nn.dropout(layer_fc2, (1.0 - dropout[1]))

        # with tf.device('/cpu:0'):
        tf.summary.histogram("weights", h2)
        tf.summary.histogram("biases", b2)
        tf.summary.histogram("activations", layer_fc2)

    # Create the forward and backward LSTM units. Inputs have length `n_cell_dim`.
    # LSTM forget gate bias initialized at `1.0` (default), meaning less forgetting
    # at the beginning of training (remembers more previous info)

    # Input shape: [batch_size, n_steps, n_input]
    # batch_x_shape = tf.shape(batch_x)

    # # # Only work on cudnn 7.0.4
    # with tf.name_scope('multilayer_lstm_cudnn'):
    #     layer_fc2 = tf.transpose(tf.reshape(layer_fc2, [batch_x_shape[0], -1, n_hidden_2]), (1,0,2))
    #     model = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers = n_layers, 
    #                                            num_units = n_cell_dim, 
    #                                            input_mode = 'linear_input',
    #                                            direction = 'unidirectional',
    #                                            dropout = dropout[2],
    #                                            kernel_initializer = tf.random_normal_initializer(stddev=0.1),
    #                                            bias_initializer = tf.random_normal_initializer(stddev=0.1))
    #     outputs, final_state = model(inputs=layer_fc2)
    #     outputs = tf.reshape(tf.transpose(outputs, (1,0,2)), [-1, n_cell_dim]) # [batch*time, n_cell_dim]
    #     tf.summary.histogram("activations", outputs)

    with tf.name_scope('multilayer_lstm'):
        # as the LSTM expects its input to be of shape `[batch_size, time, input_size]`.        
        layer_fc2 = tf.reshape(layer_fc2, [batch_x_shape[0], -1, n_hidden_2])
        # `layer_fc2` is now reshaped into `[n_steps, batch_size, n_hidden_2]`,
        cells = []
        for _ in range(n_layers):
            cell = tf.nn.rnn_cell.BasicLSTMCell(n_cell_dim, state_is_tuple = True)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = 1 - dropout[2])
            cells.append(cell)
        stack = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)        
        # Get layer activations (second output is the final state of the layer, do not need)
        outputs, final_state = tf.nn.dynamic_rnn(cell = stack, 
                                                inputs = layer_fc2, 
                                                initial_state=initial_state,
                                                sequence_length = seq_length,
                                                dtype = tf.float32,
                                                time_major=False
                                               )


        # outputs has the shape of [batch_size, time, n_cell_dim]
        # states has the shape of [batch_size, cell.state_size]
        # Reshape to apply the same weights over the timesteps
        outputs = tf.reshape(outputs, [-1, n_cell_dim]) # [batch*time, n_cell_dim]
        # with tf.device('/cpu:0'):
        tf.summary.histogram("activations", outputs)

    with tf.name_scope('fc3'):
        # Now we feed `outputs` to the fifth hidden layer with clipped RELU activation and dropout
        b3 = variable_on_gpu('b3', [n_hidden_3], tf.random_normal_initializer(stddev=b3_stddev))
        h3 = variable_on_gpu('h3', [n_cell_dim, n_hidden_3], tf.random_normal_initializer(stddev=h3_stddev))
        layer_fc3 = tf.nn.elu(tf.nn.xw_plus_b(outputs, h3, b3))
        layer_fc3 = tf.nn.dropout(layer_fc3, (1.0 - dropout[4]))
        # with tf.device('/cpu:0'):
        tf.summary.histogram("weights", h3)
        tf.summary.histogram("biases", b3)
        tf.summary.histogram("activations", layer_fc3)

    with tf.name_scope('fc4'):
        # Now we feed `outputs` to the fifth hidden layer with clipped RELU activation and dropout
        b4 = variable_on_gpu('b4', [n_hidden_4], tf.random_normal_initializer(stddev=b4_stddev))
        h4 = variable_on_gpu('h4', [n_hidden_3, n_hidden_4], tf.random_normal_initializer(stddev=h4_stddev))
        layer_fc4 = tf.nn.elu(tf.nn.xw_plus_b(layer_fc3, h4, b4))
        layer_fc4 = tf.nn.dropout(layer_fc4, (1.0 - dropout[5]))
        # with tf.device('/cpu:0'):
        tf.summary.histogram("weights", h4)
        tf.summary.histogram("biases", b4)
        tf.summary.histogram("activations", layer_fc4)

    ####################################
    ##########  LOGIT LAYER  ###########
    ####################################
    with tf.name_scope('output_layer'):
        b_out = variable_on_gpu('b_out', [n_out], tf.random_normal_initializer(stddev=b_out_stddev))
        h_out = variable_on_gpu('h_out', [n_hidden_4, n_out], tf.random_normal_initializer(stddev=h_out_stddev))
        layer_out = tf.nn.xw_plus_b(layer_fc4, h_out, b_out) # shape of [batch*time, n_out]

        tf.summary.histogram("weights", h_out)
        tf.summary.histogram("biases", b_out)
        tf.summary.histogram("activations", layer_out)
        # # shape of [batch_size*time, 1 + n_mixture * n_prob_param]
        # n_prob_param = 1 (pi) + 4 (mu) + 10 (cov)

    with tf.name_scope('mixture_coef'):
        end_layer, \
         pi_layer, \
          mu_layer, \
           L_layer = get_mixture_coef(layer_out, n_mixture)

    with tf.name_scope('mixture_distribution'):
        MVN_pdf = tf.contrib.distributions.MultivariateNormalTriL(loc = mu_layer, scale_tril = L_layer)

    summary_op = tf.summary.merge_all()
    return end_layer, pi_layer, mu_layer, L_layer, MVN_pdf, final_state, summary_op
    # return end_layer, pi_layer, mu_lat_layer, mu_lon_layer, mu_alt_layer, mu_st_layer, sigma_lat_layer, sigma_lon_layer, sigma_alt_layer, sigma_st_layer, rho_latlon_layer, rho_lonalt_layer, rho_altlat_layer, rho_stlat_layer, rho_stlon_layer, rho_stalt_layer, final_state, summary_op

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

    mu_layer = tf.reshape(mu_layer, (-1, n_mixture, 4)) # [batch*time, 4, n_mixture]
    L_layer = tf.reshape(L_layer, (-1, n_mixture, 10))
    L_layer = tf.contrib.distributions.fill_triangular(L_layer)
    L_layer = tf.contrib.distributions.matrix_diag_transform(L_layer, transform=tf.nn.softplus)
    
    return end_layer, pi_layer, mu_layer, L_layer


    # split_shape = [1]
    # split_shape.extend([n_mixture]*(n_prob_param//3))
    # split_shape.append(n_mixture * n_prob_param*2//3)
    # end_layer, \
    #  pi_layer, \
    #   mu_lat_layer, \
    #    mu_lon_layer, \
    #     mu_alt_layer, \
    #      mu_st_layer, \
    #       L_layer = tf.split(value = layer_out, 
    #                          num_or_size_splits = split_shape,
    #                          axis = 1)
    # mu_flat = tf.concat([mu_lat_layer, mu_lon_layer, mu_alt_layer, mu_st_layer], axis = 1) # [batch*time, n_mixture * 4]
    # mu_layer = tf.transpose(tf.reshape(mu_flat, (-1, 4, n_mixture)), perm = [0,2,1]) # [batch*time, 4, n_mixture]
    # sigma_alt_layer = tf.exp(sigma_alt_layer)
    # sigma_lat_layer = tf.exp(sigma_lat_layer)
    # sigma_lon_layer = tf.exp(sigma_lon_layer)
    # sigma_st_layer = tf.exp(sigma_st_layer)

    # rho_latlon_layer = tf.nn.tanh(rho_latlon_layer)
    # rho_lonalt_layer = tf.nn.tanh(rho_lonalt_layer)
    # rho_altlat_layer = tf.nn.tanh(rho_altlat_layer)
    # rho_stlat_layer = tf.nn.tanh(rho_stlat_layer)
    # rho_stlon_layer = tf.nn.tanh(rho_stlon_layer)
    # rho_stalt_layer = tf.nn.tanh(rho_stalt_layer)




    # b_end_stddev = 0.01
    # h_end_stddev = 0.01
    # b_pi_stddev = 0.01
    # h_pi_stddev = 0.01
    # b_mu_lat_stddev = 0.01
    # h_mu_lat_stddev = 0.01
    # b_mu_lon_stddev = 0.01
    # h_mu_lon_stddev = 0.01
    # b_mu_alt_stddev = 0.01
    # h_mu_alt_stddev = 0.01
    # b_sigma_lat_stddev = 0.01
    # h_sigma_lat_stddev = 0.01
    # b_sigma_lon_stddev = 0.01
    # h_sigma_lon_stddev = 0.01
    # b_sigma_alt_stddev = 0.01
    # h_sigma_alt_stddev = 0.01
    # b_rho_lat_stddev = 0.01
    # h_rho_lat_stddev = 0.01
    # b_rho_lon_stddev = 0.01
    # h_rho_lon_stddev = 0.01
    # b_rho_alt_stddev = 0.01
    # h_rho_alt_stddev = 0.01
    # with tf.name_scope('fc_end'):
    #     # Now we feed `outputs` to the fifth hidden layer with clipped RELU activation and dropout
    #     b_end = variable_on_gpu('b_end', [1], tf.random_normal_initializer(stddev=b_end_stddev))
    #     h_end = variable_on_gpu('h_end', [n_hidden_4, 1], tf.random_normal_initializer(stddev=h_end_stddev))
    #     end_layer = tf.nn.sigmoid(tf.add(tf.matmul(layer_fc4, h_end), b_end))

    #     with tf.device('/cpu:0'):
    #         tf.summary.histogram("weights", h_end)
    #         tf.summary.histogram("biases", b_end)
    #         tf.summary.histogram("activations", end_layer)

    #     end_layer = tf.reshape(end_layer, [-1, batch_x_shape[0], n_mixture])
    #     end_layer = tf.transpose(end_layer, [1, 0, 2])
    #     # shape of [batch_size, time, n_dim]

    # with tf.name_scope('fc_pi'):
    #     # Now we feed `outputs` to the fifth hidden layer with clipped RELU activation and dropout
    #     b_pi = variable_on_gpu('b_pi', [n_mixture], tf.random_normal_initializer(stddev=b_pi_stddev))
    #     h_pi = variable_on_gpu('h_pi', [n_hidden_4, n_mixture], tf.random_normal_initializer(stddev=h_pi_stddev))
    #     pi_layer = tf.nn.softmax(tf.add(tf.matmul(layer_fc4, h_pi), b_pi))

    #     with tf.device('/cpu:0'):
    #         tf.summary.histogram("weights", h_pi)
    #         tf.summary.histogram("biases", b_pi)
    #         tf.summary.histogram("activations", pi_layer)

    #     pi_layer = tf.reshape(pi_layer, [-1, batch_x_shape[0], n_mixture])
    #     pi_layer = tf.transpose(pi_layer, [1, 0, 2])
    #     # shape of [batch_size, time, n_dim]

    # ###############################################################
    # with tf.name_scope('fc_mu_lat'):
    #     # Now we feed `outputs` to the fifth hidden layer with clipped RELU activation and dropout
    #     b_mu_lat = variable_on_gpu('b_mu_lat', [n_mixture], tf.random_normal_initializer(stddev=b_mu_lat_stddev))
    #     h_mu_lat = variable_on_gpu('h_lat', [n_hidden_4, n_mixture], tf.random_normal_initializer(stddev=h_mu_lat_stddev))
    #     mu_lat_layer = (tf.add(tf.matmul(layer_fc4, h_mu_lat), b_mu_lat))

    #     with tf.device('/cpu:0'):
    #         tf.summary.histogram("weights", h_mu_lat)
    #         tf.summary.histogram("biases", b_mu_lat)
    #         tf.summary.histogram("activations", mu_lat_layer)

    #     mu_lat_layer = tf.reshape(mu_lat_layer, [-1, batch_x_shape[0], n_mixture])
    #     mu_lat_layer = tf.transpose(mu_lat_layer, [1, 0, 2])
    #     # shape of [batch_size, time, n_dim]

    # with tf.name_scope('fc_mu_lon'):
    #     # Now we feed `outputs` to the fifth hidden layer with clipped RELU activation and dropout
    #     b_mu_lon = variable_on_gpu('b_mu_lon', [n_mixture], tf.random_normal_initializer(stddev=b_mu_lon_stddev))
    #     h_mu_lon = variable_on_gpu('h_lon', [n_hidden_4, n_mixture], tf.random_normal_initializer(stddev=h_mu_lon_stddev))
    #     mu_lon_layer = (tf.add(tf.matmul(layer_fc4, h_mu_lon), b_mu_lon))

    #     with tf.device('/cpu:0'):
    #         tf.summary.histogram("weights", h_mu_lon)
    #         tf.summary.histogram("biases", b_mu_lon)
    #         tf.summary.histogram("activations", mu_lon_layer)

    #     mu_lon_layer = tf.reshape(mu_lon_layer, [-1, batch_x_shape[0], n_mixture])
    #     mu_lon_layer = tf.transpose(mu_lon_layer, [1, 0, 2])
    #     # shape of [batch_size, time, n_dim]

    # with tf.name_scope('fc_mu_alt'):
    #     # Now we feed `outputs` to the fifth hidden layer with clipped RELU activation and dropout
    #     b_mu_alt = variable_on_gpu('b_mu_alt', [n_mixture], tf.random_normal_initializer(stddev=b_mu_alt_stddev))
    #     h_mu_alt = variable_on_gpu('h_alt', [n_hidden_4, n_mixture], tf.random_normal_initializer(stddev=h_mu_alt_stddev))
    #     mu_alt_layer = (tf.add(tf.matmul(layer_fc4, h_mu_alt), b_mu_alt))

    #     with tf.device('/cpu:0'):
    #         tf.summary.histogram("weights", h_mu_alt)
    #         tf.summary.histogram("biases", b_mu_alt)
    #         tf.summary.histogram("activations", mu_alt_layer)

    #     mu_alt_layer = tf.reshape(mu_alt_layer, [-1, batch_x_shape[0], n_mixture])
    #     mu_alt_layer = tf.transpose(mu_alt_layer, [1, 0, 2])
    #     # shape of [batch_size, time, n_dim]

    # with tf.name_scope('fc_mu_st'):
    #     # Now we feed `outputs` to the fifth hidden layer with clipped RELU activation and dropout
    #     b_mu_st = variable_on_gpu('b_mu_st', [n_mixture], tf.random_normal_initializer(stddev=b_mu_st_stddev))
    #     h_mu_st = variable_on_gpu('h_st', [n_hidden_4, n_mixture], tf.random_normal_initializer(stddev=h_mu_st_stddev))
    #     mu_st_layer = (tf.add(tf.matmul(layer_fc4, h_mu_st), b_mu_st))

    #     with tf.device('/cpu:0'):
    #         tf.summary.histogram("weights", h_mu_st)
    #         tf.summary.histogram("biases", b_mu_st)
    #         tf.summary.histogram("activations", mu_st_layer)

    #     mu_st_layer = tf.reshape(mu_st_layer, [-1, batch_x_shape[0], n_mixture])
    #     mu_st_layer = tf.transpose(mu_st_layer, [1, 0, 2])
    #     # shape of [batch_size, time, n_dim]
    
    # ###############################################################
    # with tf.name_scope('fc_sigma_lat'):
    #     # Now we feed `outputs` to the fifth hidden layer with clipped RELU activation and dropout
    #     b_sigma_lat = variable_on_gpu('b_sigma_lat', [n_mixture], tf.random_normal_initializer(stddev=b_sigma_lat_stddev))
    #     h_sigma_lat = variable_on_gpu('h_lat', [n_hidden_4, n_mixture], tf.random_normal_initializer(stddev=h_sigma_lat_stddev))
    #     sigma_lat_layer = tf.nn.softplus(tf.add(tf.matmul(layer_fc4, h_sigma_lat), b_sigma_lat))

    #     with tf.device('/cpu:0'):
    #         tf.summary.histogram("weights", h_sigma_lat)
    #         tf.summary.histogram("biases", b_sigma_lat)
    #         tf.summary.histogram("activations", sigma_lat_layer)

    #     sigma_lat_layer = tf.reshape(sigma_lat_layer, [-1, batch_x_shape[0], n_mixture])
    #     sigma_lat_layer = tf.transpose(sigma_lat_layer, [1, 0, 2])
    #     # shape of [batch_size, time, n_dim]

    # with tf.name_scope('fc_sigma_lon'):
    #     # Now we feed `outputs` to the fifth hidden layer with clipped RELU activation and dropout
    #     b_sigma_lon = variable_on_gpu('b_sigma_lon', [n_mixture], tf.random_normal_initializer(stddev=b_sigma_lon_stddev))
    #     h_sigma_lon = variable_on_gpu('h_lon', [n_hidden_4, n_mixture], tf.random_normal_initializer(stddev=h_sigma_lon_stddev))
    #     sigma_lon_layer = tf.nn.softplus(tf.add(tf.matmul(layer_fc4, h_sigma_lon), b_sigma_lon))

    #     with tf.device('/cpu:0'):
    #         tf.summary.histogram("weights", h_sigma_lon)
    #         tf.summary.histogram("biases", b_sigma_lon)
    #         tf.summary.histogram("activations", sigma_lon_layer)

    #     sigma_lon_layer = tf.reshape(sigma_lon_layer, [-1, batch_x_shape[0], n_mixture])
    #     sigma_lon_layer = tf.transpose(sigma_lon_layer, [1, 0, 2])
    #     # shape of [batch_size, time, n_dim]

    # with tf.name_scope('fc_sigma_alt'):
    #     # Now we feed `outputs` to the fifth hidden layer with clipped RELU activation and dropout
    #     b_sigma_alt = variable_on_gpu('b_sigma_alt', [n_mixture], tf.random_normal_initializer(stddev=b_sigma_alt_stddev))
    #     h_sigma_alt = variable_on_gpu('h_alt', [n_hidden_4, n_mixture], tf.random_normal_initializer(stddev=h_sigma_alt_stddev))
    #     sigma_alt_layer = tf.nn.softplus(tf.add(tf.matmul(layer_fc4, h_sigma_alt), b_sigma_alt))

    #     with tf.device('/cpu:0'):
    #         tf.summary.histogram("weights", h_sigma_alt)
    #         tf.summary.histogram("biases", b_sigma_alt)
    #         tf.summary.histogram("activations", sigma_alt_layer)

    #     sigma_alt_layer = tf.reshape(sigma_alt_layer, [-1, batch_x_shape[0], n_mixture])
    #     sigma_alt_layer = tf.transpose(sigma_alt_layer, [1, 0, 2])
    #     # shape of [batch_size, time, n_dim]

    # with tf.name_scope('fc_sigma_st'):
    #     # Now we feed `outputs` to the fifth hidden layer with clipped RELU activation and dropout
    #     b_sigma_st = variable_on_gpu('b_sigma_st', [n_mixture], tf.random_normal_initializer(stddev=b_sigma_st_stddev))
    #     h_sigma_st = variable_on_gpu('h_st', [n_hidden_4, n_mixture], tf.random_normal_initializer(stddev=h_sigma_st_stddev))
    #     sigma_st_layer = tf.nn.softplus(tf.add(tf.matmul(layer_fc4, h_sigma_st), b_sigma_st))

    #     with tf.device('/cpu:0'):
    #         tf.summary.histogram("weights", h_sigma_st)
    #         tf.summary.histogram("biases", b_sigma_st)
    #         tf.summary.histogram("activations", sigma_st_layer)

    #     sigma_st_layer = tf.reshape(sigma_st_layer, [-1, batch_x_shape[0], n_mixture])
    #     sigma_st_layer = tf.transpose(sigma_st_layer, [1, 0, 2])
    #     # shape of [batch_size, time, n_dim]

    # ###############################################################
    # with tf.name_scope('fc_rho_lat'):
    #     # rho_latlon
    #     # Now we feed `outputs` to the fifth hidden layer with clipped RELU activation and dropout
    #     b_rho_lat = variable_on_gpu('b_rho_lat', [n_mixture], tf.random_normal_initializer(stddev=b_rho_lat_stddev))
    #     h_rho_lat = variable_on_gpu('h_lat', [n_hidden_4, n_mixture], tf.random_normal_initializer(stddev=h_rho_lat_stddev))
    #     rho_lat_layer = tf.nn.tanh(tf.add(tf.matmul(layer_fc4, h_rho_lat), b_rho_lat))

    #     with tf.device('/cpu:0'):
    #         tf.summary.histogram("weights", h_rho_lat)
    #         tf.summary.histogram("biases", b_rho_lat)
    #         tf.summary.histogram("activations", rho_lat_layer)

    #     rho_lat_layer = tf.reshape(rho_lat_layer, [-1, batch_x_shape[0], n_mixture])
    #     rho_lat_layer = tf.transpose(rho_lat_layer, [1, 0, 2])
    #     # shape of [batch_size, time, n_dim]

    # with tf.name_scope('fc_rho_lon'):
    #     # rho_lonalt
    #     b_rho_lon = variable_on_gpu('b_rho_lon', [n_mixture], tf.random_normal_initializer(stddev=b_rho_lon_stddev))
    #     h_rho_lon = variable_on_gpu('h_lon', [n_hidden_4, n_mixture], tf.random_normal_initializer(stddev=h_rho_lon_stddev))
    #     rho_lon_layer = tf.nn.tanh(tf.add(tf.matmul(layer_fc4, h_rho_lon), b_rho_lon))

    #     with tf.device('/cpu:0'):
    #         tf.summary.histogram("weights", h_rho_lon)
    #         tf.summary.histogram("biases", b_rho_lon)
    #         tf.summary.histogram("activations", rho_lon_layer)

    #     rho_lon_layer = tf.reshape(rho_lon_layer, [-1, batch_x_shape[0], n_mixture])
    #     rho_lon_layer = tf.transpose(rho_lon_layer, [1, 0, 2])
    #     # shape of [batch_size, time, n_dim]

    # with tf.name_scope('fc_rho_alt'):
    #     # rho_latalt
    #     b_rho_alt = variable_on_gpu('b_rho_alt', [n_mixture], tf.random_normal_initializer(stddev=b_rho_alt_stddev))
    #     h_rho_alt = variable_on_gpu('h_alt', [n_hidden_4, n_mixture], tf.random_normal_initializer(stddev=h_rho_alt_stddev))
    #     rho_alt_layer = tf.nn.tanh(tf.add(tf.matmul(layer_fc4, h_rho_alt), b_rho_alt))

    #     with tf.device('/cpu:0'):
    #         tf.summary.histogram("weights", h_rho_alt)
    #         tf.summary.histogram("biases", b_rho_alt)
    #         tf.summary.histogram("activations", rho_alt_layer)

    #     rho_alt_layer = tf.reshape(rho_alt_layer, [-1, batch_x_shape[0], n_mixture])
    #     rho_alt_layer = tf.transpose(rho_alt_layer, [1, 0, 2])
    #     # shape of [batch_size, time, n_dim]

    # # with tf.device('/cpu:0'):
    # summary_op = tf.summary.merge_all()

    # Output shape: [batch_size, n_steps, n_hidden_6]
    # return end_layer, pi_layer, mu_lat_layer, mu_lon_layer, mu_alt_layer, mu_st_layer, sigma_lat_layer, sigma_lon_layer, sigma_alt_layer, sigma_st_layer, rho_lat_layer, rho_lon_layer, rho_alt_layer, summary_op