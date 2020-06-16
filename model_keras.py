import numpy as np

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Conv1D, TimeDistributed, Flatten


class S2sModel:

    def __init__(self, **kwargs):

        self.num_encoder_tokens = kwargs.get("num_encoder_tokens", -1)

        # add BOS, EOS, EMPTY
        self.num_decoder_tokens = kwargs.get("num_decoder_tokens", -1)
        self.latent_dim = kwargs.get("latent_dim", -1)
        self.model, self.encoder_model, self.decoder_model = seq2seq(self.num_encoder_tokens,
                                                                     self.num_decoder_tokens,
                                                                     self.latent_dim)

    def predict(self, input_seq, max_decoder_seq_length=512):
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
        stop_condition = False
        output_tokens = None
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + states_value)

            # Exit condition: either hit max length
            # or find stop character.
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            if (sampled_token_index == len(output_tokens[0, -1, :]) - 1 or
                sampled_token_index == 0 or
                    len(output_tokens[0]) > max_decoder_seq_length):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return output_tokens[0]

    def fit(self, encoder_input_data, decoder_input_data, decoder_target_data, batch_size=32, epochs=12, **kwargs):
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

        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_split=0.2,
                       **kwargs)

    def load(self):
        raise NotImplemented

    def save(self, model_path):
        raise NotImplemented


def seq2seq(num_encoder_tokens, num_decoder_tokens, latent_dim, len_seq=10):
    """

    :param num_encoder_tokens: length of beat
    :param num_decoder_tokens: number of classes
    :param latent_dim:
    :return:
    """

    # Define an input sequence and process it.
    img_inputs = Input(shape=(None, num_encoder_tokens, 1))

    encoder_inputs = TimeDistributed(Conv1D(100, 3, activation="relu"))(img_inputs)
    lstm_inputs = TimeDistributed(Flatten())(encoder_inputs)
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
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([img_inputs, decoder_inputs], decoder_outputs)

    encoder_model = Model(img_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    # here is the link between decoder_model and model, using decoder_lstm
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]

    # here is the link between decoder_model and model, using decoder_dense
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    return model, encoder_model, decoder_model
