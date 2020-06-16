import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
from sklearn.preprocessing import MinMaxScaler
import random
import time
import os
import pickle
import gc

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from datetime import datetime
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import argparse

random.seed(654)


def read_mitbih(filename,
                max_time=100,
                classes=['F', 'N', 'S', 'V', 'Q'],
                max_nlabel=1000,
                do_reorder_samples=False):
    """

    :param filename:
    :param max_time: int, sequence length, or in other word, number of beats per training sample
    :param classes:
    :param max_nlabel: int, upper bound of number of samples in each class
    :param do_reorder_samples:
    :return:
    """

    # read data

    samples = spio.loadmat(filename + ".mat")
    samples = samples['s2s_mitbih']

    values_mat = samples[0]['seg_values']  # dim: record, beat, unknown?(=0), amplitude, channel
    labels_mat = samples[0]['seg_labels']  # dim: record, unknown?(=0), beat

    # let the data make more sense (remove unknow dimension)
    values = [[beat[0] for beat in sig] for sig in values_mat]
    labels = [sig_labels[0] for sig_labels in labels_mat]

    # start preprocess
    num_beats = sum([len(record) for record in values])

    # number of sequences
    n_seqs = num_beats // max_time

    #  add all segments(beats) together to the length of n_seqs * max_time (if label in classes)
    data = []
    t_labels = []
    for r in range(len(values)):
        assert len(values[r]) == len(labels[r])

        for j in range(len(values[r])):
            if len(data) == n_seqs * max_time:
                break
            elif labels[r][j] in classes:
                data.append(values[r][j])
                t_labels.append(labels[r][j])

    del values
    gc.collect()

    # ravel the data
    data = np.reshape(data, [len(data), -1])
    t_labels = np.array(t_labels)

    if do_reorder_samples:
        # ERROR: here is the problem
        _data, _labels = reorder_samples(data, t_labels, max_nlabel)
        # Note: with max_nlabel, len(_data) maybe changed, so we need cut the data again to make sure mod(len) == 0
    else:
        _data, _labels = data, t_labels

    # cut the data
    # data = _data
    data = _data[:(len(_data) // max_time) * max_time, :]
    _labels = _labels[:(len(_data) // max_time) * max_time]

    #  split data into sublist of 100=se_len values
    data = [data[i:i + max_time] for i in range(0, len(data), max_time)]
    labels = [_labels[i:i + max_time] for i in range(0, len(_labels), max_time)]

    # shuffle the sub_signal (the segment)
    data = np.asarray(data)
    labels = np.asarray(labels)

    permute = np.random.permutation(len(labels))
    data = data[permute]
    labels = labels[permute]

    print('Records processed!')

    return data, labels


def reorder_samples(x, y, upper_bound_nsamples_per_class=None):
    """
    # ERROR: so here is the logic error:
    # Both in train and test set:
    # _labels is a list like ["N", "N", ..., "N", "S", ... , "S", "V", .... "V"]
    # in reality how do you reorder the beats when you do not know the label beforehand.

    :param x: List[np.array]
    :param y: List[str]
    :param upper_bound_nsamples_per_class:
    :return:
    """

    _data = np.asarray([], dtype=np.float64).reshape(0, x.shape[1])
    _labels = np.asarray([], dtype=np.dtype('|S1')).reshape(0, )
    for cls in np.unique(y):
        indx_label = np.where(y == cls)[0]

        if upper_bound_nsamples_per_class:
            permute = np.random.permutation(len(indx_label))[:upper_bound_nsamples_per_class]
        else:
            permute = np.random.permutation(len(indx_label))

        indx_label_permutated = indx_label[permute]

        _data = np.concatenate((_data, x[indx_label_permutated]))
        _labels = np.concatenate((_labels, y[indx_label_permutated]))

    return _data, _labels


def evaluate_metrics(confusion_matrix):
    # https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)

    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)
    # ACC_micro = (sum(TP) + sum(TN)) / (sum(TP) + sum(FP) + sum(FN) + sum(TN))
    ACC_macro = np.mean(
        ACC)  # to get a sense of effectiveness of our method on the small classes we computed this average (macro-average)

    return ACC_macro, ACC, TPR, TNR, PPV


def batch_data(x, y, batch_size):
    shuffle = np.random.permutation(len(x))
    start = 0
    #     from IPython.core.debugger import Tracer; Tracer()()
    x = x[shuffle]
    y = y[shuffle]
    while start + batch_size <= len(x):
        yield x[start:start + batch_size], y[start:start + batch_size]
        start += batch_size


def build_network(inputs, dec_inputs, char2numY, n_channels=10, input_depth=280, num_units=128, max_time=10,
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
    output_embedding = tf.Variable(tf.random_uniform((len(char2numY), embed_size), -1.0, 1.0), name='dec_embedding')
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

    logits = tf.layers.dense(dec_outputs, units=len(char2numY), use_bias=True)

    return logits


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def run_program(args):
    print(args)
    max_time = args.max_time  # 5 3 second best 10# 40 # 100
    epochs = args.epochs  # 300
    batch_size = args.batch_size  # 10
    num_units = args.num_units
    bidirectional = args.bidirectional
    # lstm_layers = args.lstm_layers
    n_oversampling = args.n_oversampling
    checkpoint_dir = args.checkpoint_dir
    ckpt_name = args.ckpt_name
    test_steps = args.test_steps
    classes = args.classes
    filename = args.data_dir

    X, Y = read_mitbih(filename, max_time, classes=classes, max_nlabel=100000)  # 11000
    print("# of sequences: ", len(X))
    input_depth = X.shape[2]
    n_channels = 10
    classes = np.unique(Y)
    char2numY = dict(zip(classes, range(len(classes))))
    n_classes = len(classes)
    print('Classes: ', classes)
    for cl in classes:
        ind = np.where(classes == cl)[0][0]
        print(cl, len(np.where(Y.flatten() == cl)[0]))
    # char2numX['<PAD>'] = len(char2numX)
    # num2charX = dict(zip(char2numX.values(), char2numX.keys()))
    # max_len = max([len(date) for date in x])
    #
    # x = [[char2numX['<PAD>']]*(max_len - len(date)) +[char2numX[x_] for x_ in date] for date in x]
    # print(''.join([num2charX[x_] for x_ in x[4]]))
    # x = np.array(x)

    char2numY['<GO>'] = len(char2numY)
    num2charY = dict(zip(char2numY.values(), char2numY.keys()))

    Y = [[char2numY['<GO>']] + [char2numY[y_] for y_ in date] for date in Y]
    Y = np.array(Y)

    x_seq_length = len(X[0])
    y_seq_length = len(Y[0]) - 1

    # Placeholders
    inputs = tf.placeholder(tf.float32, [None, max_time, input_depth], name='inputs')
    targets = tf.placeholder(tf.int32, (None, None), 'targets')
    dec_inputs = tf.placeholder(tf.int32, (None, None), 'output')

    # logits = build_network(inputs,dec_inputs=dec_inputs)
    logits = build_network(inputs, dec_inputs, char2numY, n_channels=n_channels, input_depth=input_depth,
                           num_units=num_units, max_time=max_time,
                           bidirectional=bidirectional)
    # decoder_prediction = tf.argmax(logits, 2)
    # confusion = tf.confusion_matrix(labels=tf.argmax(targets, 1), predictions=tf.argmax(logits, 2), num_classes=len(char2numY) - 1)# it is wrong
    # mean_accuracy,update_mean_accuracy = tf.metrics.mean_per_class_accuracy(labels=targets, predictions=decoder_prediction, num_classes=len(char2numY) - 1)

    with tf.name_scope("optimization"):
        # Loss function
        vars = tf.trainable_variables()
        beta = 0.001
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars
                           if 'bias' not in v.name]) * beta
        loss = tf.contrib.seq2seq.sequence_loss(logits, targets, tf.ones([batch_size, y_seq_length]))
        # Optimizer
        loss = tf.reduce_mean(loss + lossL2)
        optimizer = tf.train.RMSPropOptimizer(1e-3).minimize(loss)

    # split the dataset into the training and test sets
    Y_in_num = [np.argmax(y) for y in Y]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y_in_num)

    # over-sampling: SMOTE
    X_train = np.reshape(X_train, [X_train.shape[0] * X_train.shape[1], -1])
    y_train = y_train[:, 1:].flatten()

    nums = []
    for cl in classes:
        ind = np.where(classes == cl)[0][0]
        nums.append(len(np.where(y_train.flatten() == ind)[0]))
    # ratio={0:nums[3],1:nums[1],2:nums[3],3:nums[3]} # the best with 11000 for N
    ratio = {0: n_oversampling, 1: nums[1], 2: n_oversampling, 3: n_oversampling}
    sm = SMOTE(random_state=12, sampling_strategy=ratio)
    X_train, y_train = sm.fit_sample(X_train, y_train)

    X_train = X_train[:(X_train.shape[0] // max_time) * max_time, :]
    y_train = y_train[:(X_train.shape[0] // max_time) * max_time]

    X_train = np.reshape(X_train, [-1, X_test.shape[1], X_test.shape[2]])
    y_train = np.reshape(y_train, [-1, y_test.shape[1] - 1, ])
    y_train = [[char2numY['<GO>']] + [y_ for y_ in date] for date in y_train]
    y_train = np.array(y_train)

    print('Classes in the training set: ', classes)
    for cl in classes:
        ind = np.where(classes == cl)[0][0]
        print(cl, len(np.where(y_train.flatten() == ind)[0]))
    print("------------------y_train samples--------------------")
    for ii in range(2):
        print(''.join([num2charY[y_] for y_ in list(y_train[ii + 5])]))
    print('Classes in the test set: ', classes)
    for cl in classes:
        ind = np.where(classes == cl)[0][0]
        print(cl, len(np.where(y_test.flatten() == ind)[0]))
    print("------------------y_test samples--------------------")
    for ii in range(2):
        print(''.join([num2charY[y_] for y_ in list(y_test[ii + 5])]))

    def test_model():
        # source_batch, target_batch = next(batch_data(X_test, y_test, batch_size))
        acc_track = []
        sum_test_conf = []
        y_true = []
        y_pred = []
        for batch_i, (source_batch, target_batch) in enumerate(batch_data(X_test, y_test, batch_size)):

            dec_input = np.zeros((len(source_batch), 1)) + char2numY['<GO>']
            for i in range(y_seq_length):
                batch_logits = sess.run(logits,
                                        feed_dict={inputs: source_batch, dec_inputs: dec_input})
                prediction = batch_logits[:, -1].argmax(axis=-1)
                dec_input = np.hstack([dec_input, prediction[:, None]])
            # acc_track.append(np.mean(dec_input == target_batch))
            acc_track.append(dec_input[:, 1:] == target_batch[:, 1:])
            y_true_batch = target_batch[:, 1:].flatten()
            y_pred_batch = dec_input[:, 1:].flatten()

            y_true.extend(y_true_batch)
            y_pred.extend(y_pred_batch)
            sum_test_conf.append(confusion_matrix(y_true_batch, y_pred_batch, labels=range(len(char2numY) - 1)))

        with open("y_true_aami.pkl", "wb") as f:
            pickle.dump(y_true, f)
        with open("y_pred.aami.pkl", "wb") as f:
            pickle.dump(y_pred, f)

        print("macro f1 score", f1_score(y_true, y_pred, average='macro'))
        print("micro f1 score", f1_score(y_true, y_pred, average='micro'))
        print(classification_report(y_true, y_pred,
                                    labels=range(len(char2numY) - 1),
                                    target_names=list(classes) + ['<GO>']))

        sum_test_conf = np.mean(np.array(sum_test_conf, dtype=np.float32), axis=0)

        # print('Accuracy on test set is: {:>6.4f}'.format(np.mean(acc_track)))

        # mean_p_class, accuracy_classes = sess.run([mean_accuracy, update_mean_accuracy],
        #                                           feed_dict={inputs: source_batch,
        #                                                      dec_inputs: dec_input[:, :-1],
        #                                                      targets: target_batch[:, 1:]})
        # print (mean_p_class)
        # print (accuracy_classes)
        acc_avg, acc, sensitivity, specificity, PPV = evaluate_metrics(sum_test_conf)
        print('Average Accuracy is: {:>6.4f} on test set'.format(acc_avg))
        for index_ in range(n_classes):
            print(
                "\t{} rhythm -> Sensitivity: {:1.4f}, Specificity : {:1.4f}, Precision (PPV) : {:1.4f}, Accuracy : {:1.4f}".format(
                    classes[index_],
                    sensitivity[
                        index_],
                    specificity[
                        index_], PPV[index_],
                    acc[index_]))
        print(
            "\t Average -> Sensitivity: {:1.4f}, Specificity : {:1.4f}, Precision (PPV) : {:1.4f}, Accuracy : {:1.4f}".format(
                np.mean(sensitivity), np.mean(specificity), np.mean(PPV), np.mean(acc)))
        return acc_avg, acc, sensitivity, specificity, PPV

    loss_track = []

    def count_prameters():
        print('# of Params: ', np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

    count_prameters()

    if (os.path.exists(checkpoint_dir) == False):
        os.mkdir(checkpoint_dir)
    # train the graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver = tf.train.Saver()
        print(str(datetime.now()))
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        pre_acc_avg = 0.0
        if ckpt and ckpt.model_checkpoint_path:
            # # Restore
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            # saver.restore(session, os.path.join(checkpoint_dir, ckpt_name))
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
            # or 'load meta graph' and restore weights
            # saver = tf.train.import_meta_graph(ckpt_name+".meta")
            # saver.restore(session,tf.train.latest_checkpoint(checkpoint_dir))
            test_model()
        else:

            for epoch_i in range(epochs):
                start_time = time.time()
                train_acc = []
                for batch_i, (source_batch, target_batch) in enumerate(batch_data(X_train, y_train, batch_size)):
                    _, batch_loss, batch_logits = sess.run([optimizer, loss, logits],
                                                           feed_dict={inputs: source_batch,
                                                                      dec_inputs: target_batch[:, :-1],
                                                                      targets: target_batch[:, 1:]})
                    loss_track.append(batch_loss)
                    train_acc.append(batch_logits.argmax(axis=-1) == target_batch[:, 1:])
                # mean_p_class,accuracy_classes = sess.run([mean_accuracy,update_mean_accuracy],
                #                         feed_dict={inputs: source_batch,
                #                                               dec_inputs: target_batch[:, :-1],
                #                                               targets: target_batch[:, 1:]})

                # accuracy = np.mean(batch_logits.argmax(axis=-1) == target_batch[:,1:])
                accuracy = np.mean(train_acc)
                print(
                    'Epoch {:3} Loss: {:>6.3f} Accuracy: {:>6.4f} Epoch duration: {:>6.3f}s'.format(epoch_i, batch_loss,
                                                                                                    accuracy,
                                                                                                    time.time() - start_time))

                if epoch_i % test_steps == 0:
                    acc_avg, acc, sensitivity, specificity, PPV = test_model()

                    print('loss {:.4f} after {} epochs (batch_size={})'.format(loss_track[-1], epoch_i + 1, batch_size))
                    save_path = os.path.join(checkpoint_dir, ckpt_name)
                    saver.save(sess, save_path)
                    print("Model saved in path: %s" % save_path)

                    # if np.nan_to_num(acc_avg) > pre_acc_avg:  # save the better model based on the f1 score
                    #     print('loss {:.4f} after {} epochs (batch_size={})'.format(loss_track[-1], epoch_i + 1, batch_size))
                    #     pre_acc_avg = acc_avg
                    #     save_path =os.path.join(checkpoint_dir, ckpt_name)
                    #     saver.save(sess, save_path)
                    #     print("The best model (till now) saved in path: %s" % save_path)

            # plt.plot(loss_track)
            # plt.show()
        print(str(datetime.now()))
        # test_model()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--max_time', type=int, default=10)
    parser.add_argument('--test_steps', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--data_dir', type=str, default='data/s2s_mitbih_aami')
    parser.add_argument('--bidirectional', type=str2bool, default=str2bool('False'))
    # parser.add_argument('--lstm_layers', type=int, default=2)
    parser.add_argument('--num_units', type=int, default=128)
    parser.add_argument('--n_oversampling', type=int, default=10000)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints-seq2seq_true')
    parser.add_argument('--ckpt_name', type=str, default='seq2seq_mitbih.ckpt')
    parser.add_argument('--classes', nargs='+', type=chr,
                        default=['F', 'N', 'S', 'V'])
    args = parser.parse_args()
    run_program(args)


if __name__ == '__main__':
    main()
