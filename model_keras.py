import numpy as np

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Conv1D, TimeDistributed, Flatten, MaxPool1D
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import RMSprop


class S2sModel:

    def __init__(self, **kwargs):

        self.num_encoder_tokens = kwargs.get("num_encoder_tokens", None)

        # add BOS, EOS, EMPTY
        self.num_decoder_tokens = kwargs.get("num_decoder_tokens", None)
        self.latent_dim = kwargs.get("latent_dim", 128)
        self.model, self.encoder_model, self.decoder_model = seq2seq(self.num_encoder_tokens,
                                                                     self.num_decoder_tokens,
                                                                     self.latent_dim)

    def predict(self, input_seq, max_decoder_seq_length=10):
        """

        :param input_seq:
        :param max_decoder_seq_length:
        :return:
        """

        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, 0] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        output = []
        while True:
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + states_value)

            # Exit condition: either hit max length
            # or find stop character.
            sampled_token_index = np.argmax(output_tokens[0, -1, :])

            # if (sampled_token_index == len(output_tokens[0, -1, :]) - 1 or
            #    sampled_token_index == 0 or
            #        len(output_tokens[0]) > max_decoder_seq_length):
            if len(output) >= max_decoder_seq_length:
                break

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

            output.append(output_tokens[0])

        return np.array(output)

    def fit(self, encoder_input_data, decoder_input_data, decoder_target_data, batch_size=10, epochs=12, **kwargs):
        """

        in the decoder_targe_data, each sequence must start with <BOS>, which is indicated by y[0] = 1.
        each sequence must end with <EOS>, which is indicated by y[-1] = 1.

        :param encoder_input_data:
        :param decoder_input_data:
        :param decoder_target_data:
        :param batch_size:
        :param epochs:
        :param kwargs:
        :return:
        """

        rmsprop = RMSprop(learning_rate=1e-3)
        self.model.compile(optimizer=rmsprop, loss='categorical_crossentropy',
                           metrics=['accuracy'])
        callbacks_list = [ModelCheckpoint("keras_model", monitor='val_loss', save_best_only=False),
                          EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto',
                                        baseline=None, restore_best_weights=False),
                          ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, verbose=0, mode='auto')
                          # TrainValTensorBoard(write_graph=False)
                          ]
        self.model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_split=0.1,
                       callbacks=callbacks_list,
                       **kwargs)

    def load(self):
        raise NotImplemented

    def save(self, model_path):
        raise NotImplemented


def seq2seq(num_encoder_tokens, num_decoder_tokens, latent_dim, n_channel=10, len_seq=10):
    """

    :param num_encoder_tokens: length of beat
    :param num_decoder_tokens: number of classes
    :param latent_dim:
    :return:
    """

    # Define an input sequence and process it.
    sig_inputs = Input(shape=(None, num_encoder_tokens // n_channel, n_channel))
    # img_inputs = Reshape((None, num_encoder_tokens//n_channel, n_channel))(sig_inputs)

    x = TimeDistributed(Conv1D(32, 2, activation="relu", padding="same"))(sig_inputs)
    x = TimeDistributed(MaxPool1D(2))(x)
    x = TimeDistributed(Conv1D(64, 2, activation="relu", padding="same"))(x)
    x = TimeDistributed(MaxPool1D(2))(x)
    cnn_outputs = TimeDistributed(Conv1D(128, 2, activation="relu", padding="same"))(x)

    lstm_inputs = TimeDistributed(Flatten())(cnn_outputs)
    encoder_outputs, state_h, state_c = LSTM(latent_dim, return_state=True)(lstm_inputs)

    # the channel between encoder and decoder.
    # the encoder output.
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    output_embedding = TimeDistributed(Dense(10))
    decoder_inputs_embed = output_embedding(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs_embed,
                                         initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([sig_inputs, decoder_inputs], decoder_outputs)

    encoder_model = Model(sig_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    # here is the link between decoder_model and model, using decoder_lstm
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs_embed, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]

    # here is the link between decoder_model and model, using decoder_dense
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    return model, encoder_model, decoder_model
