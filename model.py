import tensorflow as tf


def birnn(inputs, dec_inputs, output_dim, n_channels=10, input_depth=280, num_units=128, max_time=10,
          bidirectional=False):

    _inputs = tf.reshape(inputs, [-1, n_channels, input_depth // n_channels])
    # _inputs = tf.reshape(inputs, [-1,input_depth,n_channels])

    # #(batch*max_time, 280, 1) --> (N, 280, 18)
    conv1 = tf.layers.conv1d(inputs=_inputs, filters=32, kernel_size=2, strides=1,
                             padding='same', activation=tf.nn.relu)
    max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='same')

    conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=64, kernel_size=2, strides=1,
                             padding='same', activation=tf.nn.relu)
    max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same')

    conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=128, kernel_size=2, strides=1,
                             padding='same', activation=tf.nn.relu)

    shape = conv3.get_shape().as_list()
    data_input_embed = tf.reshape(conv3, (-1, max_time, shape[1] * shape[2]))

    # timesteps = max_time
    #
    # lstm_in = tf.unstack(data_input_embed, timesteps, 1)
    # lstm_size = 128
    # # Get lstm cell output
    # # Add LSTM layers
    # lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    # data_input_embed, states = tf.contrib.rnn.static_rnn(lstm_cell, lstm_in, dtype=tf.float32)
    # data_input_embed = tf.stack(data_input_embed, 1)
    # shape = data_input_embed.get_shape().as_list()

    embed_size = 10  # 128 lstm_size # shape[1]*shape[2]

    # Embedding layers
    output_embedding = tf.Variable(tf.random_uniform((output_dim, embed_size), -1.0, 1.0), name='dec_embedding')
    data_output_embed = tf.nn.embedding_lookup(output_embedding, dec_inputs)

    with tf.variable_scope("encoding") as encoding_scope:
        if not bidirectional:

            # Regular approach with LSTM units
            lstm_enc = tf.contrib.rnn.LSTMCell(num_units)
            _, last_state = tf.nn.dynamic_rnn(lstm_enc, inputs=data_input_embed, dtype=tf.float32)

        else:

            # Using a bidirectional LSTM architecture instead
            enc_fw_cell = tf.contrib.rnn.LSTMCell(num_units)
            enc_bw_cell = tf.contrib.rnn.LSTMCell(num_units)

            ((enc_fw_out, enc_bw_out), (enc_fw_final, enc_bw_final)) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=enc_fw_cell,
                cell_bw=enc_bw_cell,
                inputs=data_input_embed,
                dtype=tf.float32)
            enc_fin_c = tf.concat((enc_fw_final.c, enc_bw_final.c), 1)
            enc_fin_h = tf.concat((enc_fw_final.h, enc_bw_final.h), 1)
            last_state = tf.contrib.rnn.LSTMStateTuple(c=enc_fin_c, h=enc_fin_h)

    with tf.variable_scope("decoding") as decoding_scope:
        if not bidirectional:
            lstm_dec = tf.contrib.rnn.LSTMCell(num_units)
        else:
            lstm_dec = tf.contrib.rnn.LSTMCell(2 * num_units)

        dec_outputs, _ = tf.nn.dynamic_rnn(lstm_dec, inputs=data_output_embed, initial_state=last_state)

    logits = tf.layers.dense(dec_outputs, units=output_dim, use_bias=True)

    return logits
