from keras import backend as K
from keras.models import load_model, Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding, LSTM, Lambda, ThresholdedReLU
from keras.layers import GlobalMaxPooling1D, AlphaDropout
from keras.layers.merge import Concatenate
from keras.losses import categorical_crossentropy
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from keras import regularizers
import numpy as np
from sklearn.utils import shuffle
from TemporalMeanPooling import TemporalMeanPooling
import pandas as pd
from collections import Counter
import itertools
from keras.utils import to_categorical
import gensim
from preTrainedReader import PreTrainedVectorReader as ptvr
from scipy.sparse import csr_matrix, lil_matrix, save_npz, load_npz
from keras.optimizers import Adam
from keras.callbacks import Callback
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

import os,sys,pickle

def custom_categorical_crossentropy(args):
    '''Calculates the cross-entropy value for multiclass classification
    problems. Note: Expects a binary class matrix instead of a vector
    of scalar classes.
    return non averaged results
    '''
    y_true, y_pred = args
    return K.mean(K.categorical_crossentropy(y_pred, y_true))

class LogEpochPerformance(Callback):
    def __init__(self, test_data, test_label, filename, every_n_epoch):
        # the test data should be a dictionary
        self.test_data = test_data
        self.test_label = test_label
        self.filename = filename
        self.every_n = every_n_epoch

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.every_n == 0:
            preds = self.model.predict(self.test_data)
            pred_labels = np.argmax(preds, axis=1)
            test_labels = self.test_label

            acc = accuracy_score(test_labels, pred_labels)
            pos_prec = precision_score(test_labels, pred_labels, pos_label=1)
            neg_prec = precision_score(test_labels, pred_labels, pos_label=0)
            pos_recall = recall_score(test_labels, pred_labels, pos_label=1)
            neg_recall = recall_score(test_labels, pred_labels, pos_label=0)
            pos_f1 = f1_score(test_labels, pred_labels, pos_label=1)
            neg_f1 = f1_score(test_labels, pred_labels, pos_label=0)
            c_matrix = confusion_matrix(test_labels, pred_labels)
            fpr, tpr, thres = roc_curve(test_labels, preds[:,1], pos_label=1)
            roc = [np.round(fpr, decimals=5), np.round(tpr, decimals=5), np.round(thres, decimals=5)]
            roc_auc = roc_auc_score(test_labels, preds[:,1], average='weighted')
            prec_pr, recall_pr, thres_pr = precision_recall_curve(test_labels, preds[:,1], pos_label=1)
            pr_curve = [np.round(prec_pr, decimals=5), np.round(recall_pr, decimals=5), np.round(thres_pr, decimals=5)]

            #_filename = self.filename + "epoch_{}.log".format(epoch)
            _filename = self.filename
            with open(_filename, 'a', encoding='utf-8') as outf:
                outf.write("epoch:\t{}\n".format(epoch))
                outf.write("accuracy:\t{}\n".format(acc))
                outf.write("pos_precision:\t{}\n".format(pos_prec))
                outf.write("neg_precision:\t{}\n".format(neg_prec))
                outf.write("pos_recall:\t{}\n".format(pos_recall))
                outf.write("neg_recall:\t{}\n".format(neg_recall))
                outf.write("pos_f1:\t{}\n".format(pos_f1))
                outf.write("neg_f1:\t{}\n".format(neg_f1))
                outf.write("roc_auc:\t{}\n".format(roc_auc))
                outf.write("c_matrix:\t{}\n".format(c_matrix.tolist()))
                outf.write("roc_curve:\t{}\n".format([x.tolist() for x in roc]))
                outf.write("pr_curve:\t{}\n".format([x.tolist() for x in pr_curve]))
            print("epoch:        {}\naccuracy:     {}\npos_prec:     {}\tneg_prec:     {}\npos_recall:   {}\tneg_recall:   {}\npos_f1:       {}\tneg_f1:       {}\nroc_auc:       {}".format(epoch, acc, pos_prec, neg_prec, pos_recall, neg_recall, pos_f1, neg_f1, roc_auc))


class PredictOnEpoch(Callback):
    def __init__(self, test_data, test_labels, every_n_epoch, epoch_results, epoch_preds):
        # the test data should be a dictionary
        self.test_data = test_data
        self.test_label = test_labels
        self.every_n = every_n_epoch
        self.epoch_results = epoch_results
        self.epoch_preds = epoch_preds

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.every_n == 0:
            preds = self.model.predict(self.test_data)
            pred_labels = np.argmax(preds, axis=1)
            test_labels = self.test_label

            acc = accuracy_score(test_labels, pred_labels)
            pos_prec = precision_score(test_labels, pred_labels, pos_label=1)
            neg_prec = precision_score(test_labels, pred_labels, pos_label=0)
            pos_recall = recall_score(test_labels, pred_labels, pos_label=1)
            neg_recall = recall_score(test_labels, pred_labels, pos_label=0)
            pos_f1 = f1_score(test_labels, pred_labels, pos_label=1)
            neg_f1 = f1_score(test_labels, pred_labels, pos_label=0)
            roc_auc = roc_auc_score(test_labels, preds[:,1], average='weighted')
            print("epoch:        {}\naccuracy:     {}\npos_prec:     {}\tneg_prec:     {}\npos_recall:   {}\tneg_recall:   {}\npos_f1:       {}\tneg_f1:       {}\nroc_auc:       {}".format(epoch, acc, pos_prec, neg_prec, pos_recall, neg_recall, pos_f1, neg_f1, roc_auc))
            
            self.epoch_results.append(pred_labels)
            self.epoch_preds.append(preds)
            # with open(self.filename, 'a+') as outf:
            #     outf.write('{}\n'.format(pred_labels))
            # print(pred_labels)

#===========================================
# LSTM model by Keras
def model_lstm_relu_embedding(word_matrix, LSTM_params, wordvec_params):
    model = Sequential() # try with the simple sequential model
    # use keras's own embedding implementation
    if LSTM_params["use_keras_embedding"]:
        model.add(Embedding(input_dim=word_matrix.shape[0], output_dim=wordvec_params['embedding_dim'],
            mask_zero=True, embeddings_regularizer=None, input_length=LSTM_params['sequence_length']))
    # use word2vec or other word-vectors
    else:
        model.add(Embedding(input_dim=word_matrix.shape[0], output_dim=wordvec_params['embedding_dim'],
            mask_zero=True, trainable=False, weights=[word_matrix],
            embeddings_regularizer=None, input_length=LSTM_params['sequence_length']))

    model.add(Dropout(LSTM_params["dropout_prob"][0]))
    # the embedding layer that maps word id into word vector
    if LSTM_params['sequence']:
        model.add(LSTM(units=LSTM_params['sequence_length'], return_sequences=True)) # the LSTM layer
        model.add(TemporalMeanPooling()) # mean pooling that takes masking into account, also removes masking
    else:
        model.add(LSTM(units=LSTM_params['sequence_length'], return_sequences=False)) # the LSTM layer
    #model.add()
    model.add(Dropout(LSTM_params["dropout_prob"][1])) # dropout layer for logisitc regression layer
    model.add(Dense(LSTM_params["hidden_dim"], activation="relu"))
    model.add(Dense(2, activation='softmax')) # logistic regression layer
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    #print("Output shape after mean pooling: " + str(model.layers[2].output_shape))
    return model

#===========================================================

# CNN model by Keras
def model_conv2d_relu_embedding(word_matrix, CONV_params, wordvec_params):
    model_input = Input(shape=[CONV_params["sequence_length"],])
    if CONV_params["use_keras_embedding"]:
        z = Embedding(input_dim=word_matrix.shape[0], output_dim=wordvec_params['embedding_dim'],
            mask_zero=False, trainable=True, embeddings_regularizer=None, 
            input_length=CONV_params['sequence_length'], name="embedding")(model_input)
    else:
        z = Embedding(input_dim=word_matrix.shape[0], output_dim=wordvec_params['embedding_dim'],
                mask_zero=False, trainable=False, weights=[word_matrix],
                input_length=CONV_params['sequence_length'], name="embedding")(model_input)
    z = Dropout(CONV_params["dropout_prob"][0])(z)
    print(z)
    conv_blocks = []
    for n_gram in CONV_params["n_gram_list"]:
        print("{} gram".format(n_gram))
        conv = Convolution1D(filters=CONV_params["num_filters"],
                            kernel_size=n_gram,
                            padding="valid",
                            activation="relu",
                            strides=1)(z)
        print(conv)
        conv = MaxPooling1D(pool_size=2)(conv)
        print(conv)
        conv = Flatten()(conv)
        print(conv)
        conv_blocks.append(conv)
    z = Concatenate()(conv_blocks)
    print(z)
    z = Dropout(CONV_params["dropout_prob"][1])(z)
    z = Dense(CONV_params["hidden_dim"], activation="relu")(z)
    model_output = Dense(2, activation="softmax")(z)
    model = Model(model_input, model_output)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["mae", "categorical_accuracy"])
    print(model.summary())
    return model

# CNN model by Keras
def model_conv2d_relu_embedding_weighted(word_matrix, CONV_params, wordvec_params):
    model_input = Input(shape=[CONV_params["sequence_length"],])
    if CONV_params["use_keras_embedding"]:
        z = Embedding(input_dim=word_matrix.shape[0], output_dim=wordvec_params['embedding_dim'],
            mask_zero=False, trainable=True, embeddings_regularizer=None, 
            input_length=CONV_params['sequence_length'], name="embedding")(model_input)
    else:
        z = Embedding(input_dim=word_matrix.shape[0], output_dim=wordvec_params['embedding_dim'],
                mask_zero=False, trainable=False, weights=[word_matrix],
                input_length=CONV_params['sequence_length'], name="embedding")(model_input)
    z = Dropout(CONV_params["dropout_prob"][0])(z)
    conv_blocks = []
    for n_gram in CONV_params["n_gram_list"]:
        print("{} gram".format(n_gram))
        conv = Convolution1D(filters=CONV_params["num_filters"],
                            kernel_size=n_gram,
                            padding="valid",
                            activation="relu",
                            strides=1)(z)
        print(conv)
        conv = MaxPooling1D(pool_size=2)(conv)
        print(conv)
        conv = Flatten()(conv)
        print(conv)
        conv_blocks.append(conv)
    z = Concatenate()(conv_blocks)
    print(z)
    z = Dropout(CONV_params["dropout_prob"][1])(z)
    z = Dense(CONV_params["hidden_dim"], activation="relu")(z)
    model_output = Dense(2, activation="softmax")(z)
    model = Model(model_input, model_output)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["mae", "categorical_accuracy"])
    return model

#======================================================

# CNN model by Keras
def model_deep_cnn_word(word_matrix, CONV_deep_params, wordvec_params):
    main_input = Input(shape=[CONV_deep_params["sequence_length"],], name='main_input')
    if CONV_deep_params["use_keras_embedding"]:
        z = Embedding(input_dim=word_matrix.shape[0], output_dim=wordvec_params['embedding_dim'],
            mask_zero=False, trainable=True, embeddings_regularizer=None, 
            input_length=CONV_deep_params['sequence_length'], name="embedding")(main_input)
    else:
        z = Embedding(input_dim=word_matrix.shape[0], output_dim=wordvec_params['embedding_dim'],
                mask_zero=False, trainable=False, weights=[word_matrix],
                input_length=CONV_deep_params['sequence_length'], name="embedding")(main_input)
    for cl in CONV_deep_params['conv_layers']:
        z = Convolution1D(filters=cl[0],
                            kernel_size=cl[1],
                            padding="valid",
                            activation="relu",
                            strides=1)(z)
        if cl[2] != -1:
            z = MaxPooling1D(pool_size=cl[2])(z)
        z = Dropout(CONV_deep_params['cnn_dropout'])(z)
    z = Flatten()(z)
    for fc in CONV_deep_params['fc_layers']:
        z = Dense(fc[0], activation="relu")(z)
        z = Dropout(fc[1])(z)
    model_output = Dense(2, activation="softmax")(z)
    model = Model(main_input, model_output)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(decay=0.05), metrics=["mae", "categorical_accuracy"])
    # print(model.summary())
    return model

# CNN model by Keras
def model_wide_cnn_word(word_matrix, CONV_wide_params, wordvec_params):
    main_input = Input(shape=[CONV_wide_params["sequence_length"],], name='main_input')
    if CONV_wide_params["use_keras_embedding"]:
        z = Embedding(input_dim=word_matrix.shape[0], output_dim=wordvec_params['embedding_dim'],
            mask_zero=False, trainable=True, embeddings_regularizer=None, 
            input_length=CONV_wide_params['sequence_length'], name="embedding")(main_input)
    else:
        z = Embedding(input_dim=word_matrix.shape[0], output_dim=wordvec_params['embedding_dim'],
                mask_zero=False, trainable=False, weights=[word_matrix],
                input_length=CONV_wide_params['sequence_length'], name="embedding")(main_input)
    conv_blocks = []
    for n_gram in CONV_wide_params["n_gram_list"]:
        conv = Convolution1D(filters=CONV_wide_params["num_filters"],
                            kernel_size=n_gram,
                            padding="valid",
                            activation="relu",
                            strides=1)(z)
        conv = MaxPooling1D(pool_size=2)(conv)
        conv = Dropout(CONV_wide_params['cnn_dropout'])(conv)
        conv = Flatten()(conv)
        conv_blocks.append(conv)
    z = Concatenate()(conv_blocks)
    for fc in CONV_wide_params['fc_layers']:
        z = Dense(fc[0], activation="relu")(z)
        z = Dropout(fc[1])(z)
    model_output = Dense(2, activation="softmax")(z)
    model = Model(main_input, model_output)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(decay=0.05), metrics=["mae", "categorical_accuracy"])
    # print(model.summary())
    return model

# CNN model by Keras
def model_wide_deep_cnn_word(word_matrix, CONV_wide_deep_params, wordvec_params):
    main_input = Input(shape=[CONV_wide_deep_params["sequence_length"],], name='main_input')
    #y_true = Input(shape=[2,], name='y_true')
    if CONV_wide_deep_params["use_keras_embedding"]:
        z = Embedding(input_dim=word_matrix.shape[0], output_dim=wordvec_params['embedding_dim'],
            mask_zero=False, trainable=True, embeddings_regularizer=None, 
            input_length=CONV_wide_deep_params['sequence_length'], name="embedding")(main_input)
    else:
        z = Embedding(input_dim=word_matrix.shape[0], output_dim=wordvec_params['embedding_dim'],
                mask_zero=False, trainable=False, weights=[word_matrix],
                input_length=CONV_wide_deep_params['sequence_length'], name="embedding")(main_input)
    conv_blocks = []
    for cl in CONV_wide_deep_params['cnn_layers']:
        conv = Convolution1D(filters=cl["num_filters"][0],
                            kernel_size=cl['ngram'],
                            padding="valid",
                            activation="relu",
                            strides=1)(z)
        conv = MaxPooling1D(pool_size=cl['pool_size'][0])(conv)
        conv = Dropout(CONV_wide_deep_params['cnn_dropout'])(conv)
        conv = Convolution1D(filters=cl["num_filters"][0],
                            kernel_size=cl['ngram'],
                            padding="valid",
                            activation="relu",
                            strides=1)(conv)
        conv = MaxPooling1D(pool_size=cl['pool_size'][0])(conv)
        conv = Dropout(CONV_wide_deep_params['cnn_dropout'])(conv)
        conv = Flatten()(conv)
        conv_blocks.append(conv)
    z = Concatenate()(conv_blocks)
    for fc in CONV_wide_deep_params['fc_layers']:
        z = Dense(fc[0], activation="relu")(z)
        z = Dropout(fc[1])(z)
    model_output = Dense(2, activation="softmax")(z)
    #loss = categorical_crossentropy(y_true, model_output)
    #loss = Lambda(custom_categorical_crossentropy, output_shape=(1,), name='custom_loss')([y_true, model_output])
    #model = Model([main_input, y_true], model_output)
    model = Model(main_input, model_output)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(decay=0.05), metrics=["mae", "categorical_accuracy"])
    print(model.summary())
    return model


# CNN model by Keras
def model_wide_deep_cnn_word_extra_feature_concat(word_matrix, CONV_params, wordvec_params):
    main_input = Input(shape=[CONV_params["sequence_length"],], name='main_input')
    term_feature = Input(shape=[CONV_params['term_feature_len'],], name='term_feature')
    cluster_feature = Input(shape=[CONV_params['cluster_feature_len'],], name='cluster_feature')

    if CONV_params["use_keras_embedding"]:
        z = Embedding(input_dim=word_matrix.shape[0], output_dim=wordvec_params['embedding_dim'],
            mask_zero=False, trainable=True, embeddings_regularizer=None, 
            input_length=CONV_params['sequence_length'], name="embedding")(main_input)
    else:
        z = Embedding(input_dim=word_matrix.shape[0], output_dim=wordvec_params['embedding_dim'],
                mask_zero=False, trainable=False, weights=[word_matrix],
                input_length=CONV_params['sequence_length'], name="embedding")(main_input)
    
    conv_blocks = []
    for cl in CONV_params['cnn_layers']:
        conv = Convolution1D(filters=cl["num_filters"][0],
                            kernel_size=cl['ngram'],
                            padding="valid",
                            activation="relu",
                            strides=1)(z)
        conv = MaxPooling1D(pool_size=cl['pool_size'][0])(conv)
        conv = Dropout(CONV_params['cnn_dropout'])(conv)
        conv = Convolution1D(filters=cl["num_filters"][1],
                            kernel_size=cl['ngram'],
                            padding="valid",
                            activation="relu",
                            strides=1)(conv)
        conv = MaxPooling1D(pool_size=cl['pool_size'][1])(conv)
        conv = Dropout(CONV_params['cnn_dropout'])(conv)
        conv = Flatten()(conv)
        conv_blocks.append(conv)
    
    z = Concatenate()(conv_blocks)
    for fc in CONV_params['fc_layers']:
        z = Dense(fc[0], activation="relu")(z)
        z = Dropout(fc[1])(z)
    pre_output = Concatenate()([z, term_feature, cluster_feature])
    model_output = Dense(2, activation="softmax")(pre_output)
    model = Model([main_input, term_feature, cluster_feature], model_output)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(decay=0.05, lr=0.0005), metrics=["mae", "categorical_accuracy", "loss"])
    # print(model.summary())
    return model

# CNN model by Keras
def model_wide_deep_cnn_word_extra_feature_parallel(word_matrix, CONV_params, wordvec_params):
    main_input = Input(shape=[CONV_params["sequence_length"],], name='main_input')
    syn_expan_input = Input(shape=[CONV_params["syn_expan_length"],], name='syn_expan')
    term_feature = Input(shape=[CONV_params['term_feature_len'],], name='term_feature')
    cluster_feature = Input(shape=[CONV_params['cluster_feature_len'],], name='cluster_feature')

    if CONV_params["use_keras_embedding"]:
        z = Embedding(input_dim=word_matrix.shape[0], output_dim=wordvec_params['embedding_dim'],
            mask_zero=False, trainable=True, embeddings_regularizer=None, 
            input_length=CONV_params['sequence_length'], name="embedding")(main_input)
    else:
        z = Embedding(input_dim=word_matrix.shape[0], output_dim=wordvec_params['embedding_dim'],
                mask_zero=False, trainable=False, weights=[word_matrix],
                input_length=CONV_params['sequence_length'], name="embedding")(main_input)

    if CONV_params["use_keras_embedding"]:
        z2 = Embedding(input_dim=word_matrix.shape[0], output_dim=wordvec_params['embedding_dim'],
            mask_zero=False, trainable=True, embeddings_regularizer=None, 
            input_length=CONV_params['syn_expan_length'], name="embedding_expan")(syn_expan_input)
    else:
        z2 = Embedding(input_dim=word_matrix.shape[0], output_dim=wordvec_params['embedding_dim'],
                mask_zero=False, trainable=False, weights=[word_matrix],
                input_length=CONV_params['syn_expan_length'], name="embedding_expan")(syn_expan_input)
    
    conv_blocks = []
    for cl in CONV_params['cnn_layers']:
        conv = Convolution1D(filters=cl["num_filters"][0],
                            kernel_size=cl['ngram'],
                            padding="valid",
                            activation="relu",
                            strides=1)(z)
        conv = MaxPooling1D(pool_size=cl['pool_size'][0])(conv)
        conv = Dropout(CONV_params['cnn_dropout'])(conv)
        conv = Convolution1D(filters=cl["num_filters"][1],
                            kernel_size=cl['ngram'],
                            padding="valid",
                            activation="relu",
                            strides=1)(conv)
        conv = MaxPooling1D(pool_size=cl['pool_size'][1])(conv)
        conv = Dropout(CONV_params['cnn_dropout'])(conv)
        conv = Flatten()(conv)
        conv_blocks.append(conv)

    for cl in CONV_params['syn_cnn_layers']:
        conv = Convolution1D(filters=cl["num_filters"][0],
                            kernel_size=cl['ngram'],
                            padding="valid",
                            activation="relu",
                            strides=1)(z2)
        conv = MaxPooling1D(pool_size=cl['pool_size'][0])(conv)
        conv = Dropout(CONV_params['cnn_dropout'])(conv)
        conv = Convolution1D(filters=cl["num_filters"][1],
                            kernel_size=cl['ngram'],
                            padding="valid",
                            activation="relu",
                            strides=1)(conv)
        conv = MaxPooling1D(pool_size=cl['pool_size'][1])(conv)
        conv = Dropout(CONV_params['cnn_dropout'])(conv)
        conv = Flatten()(conv)
        conv_blocks.append(conv)
    
    z = Concatenate()(conv_blocks)
    for fc in CONV_params['fc_layers']:
        z = Dense(fc[0], activation="relu")(z)
        z = Dropout(fc[1])(z)
    pre_output = Concatenate()([z, term_feature, cluster_feature])
    model_output = Dense(2, activation="softmax")(pre_output)
    model = Model([main_input, syn_expan_input, term_feature, cluster_feature], model_output)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(decay=0.05), metrics=["mae", "categorical_accuracy"])
    # print(model.summary())
    return model

#==============================================

# CNN model by Keras
def model_deep_cnn_char(CONV_char_deep_params):
    main_input = Input(shape=[CONV_char_deep_params["sequence_length"],], dtype='int64', name='main_input')
    z = Embedding(input_dim=CONV_char_deep_params["vocab_size"] + 1, output_dim=CONV_char_deep_params['embedding_size'], input_length=CONV_char_deep_params['sequence_length'], trainable=True)(main_input)
    # z = Dropout(CONV_params["dropout_prob"][0])(z)
    for cl in CONV_char_deep_params["conv_layers"]:
        z = Convolution1D(filters=cl[0], kernel_size=cl[1], padding="valid", activation=None)(z)
        z = ThresholdedReLU(CONV_char_deep_params["relu_threshold"])(z)
        if cl[2] != -1:
            z = MaxPooling1D(pool_size=cl[2])(z)
        z = Dropout(CONV_char_deep_params['cnn_dropout'])(z)
    z = Flatten()(z)
    for fc in CONV_char_deep_params["fc_layers"]:
        z = Dense(fc[0])(z)
        z = ThresholdedReLU(CONV_char_deep_params["relu_threshold"])(z)
        z = Dropout(fc[1])(z)
    model_output = Dense(2, activation="softmax")(z)
    model = Model(inputs=main_input, outputs=model_output)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(decay=0.02), metrics=["mae", "categorical_accuracy"])
    # print(model.summary())
    return model

# CNN model by Keras
def model_wide_cnn_char(CONV_char_wide_params):
    main_input = Input(shape=[CONV_char_wide_params["sequence_length"],], dtype='int64', name='main_input')
    # Embedding layers
    z = Embedding(input_dim=CONV_char_wide_params["vocab_size"] + 1, output_dim=CONV_char_wide_params['embedding_size'], input_length=CONV_char_wide_params['sequence_length'], trainable=True)(main_input)
    # Convolution layers
    convolution_output = []
    for num_filters, filter_width in CONV_char_wide_params["conv_layers"]:
        conv = Convolution1D(filters=num_filters, kernel_size=filter_width, activation='tanh', name='Conv1D_{}_{}'.format(num_filters, filter_width))(z)
        pool = GlobalMaxPooling1D(name='MaxPoolingOverTime_{}_{}'.format(num_filters, filter_width))(conv)
        convolution_output.append(pool)
    z = Concatenate()(convolution_output)
    # Fully connected layers
    for fc in CONV_char_wide_params["fc_layers"]:
        z = Dense(fc[0], activation='selu', kernel_initializer='lecun_normal')(z)
        z = AlphaDropout(fc[1])(z)
    model_output = Dense(2, activation="softmax")(z)
    model = Model(main_input, model_output)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(decay=0.02, lr=0.0005), metrics=["mae", "categorical_accuracy"])
    # print(model.summary())
    return model

# CNN model by Keras
def model_wide_cnn_char_extra_feature(CONV_char_wide_params):
    main_input = Input(shape=[CONV_char_wide_params["sequence_length"],], dtype='int64', name='main_input')
    term_feature = Input(shape=[CONV_char_wide_params['term_feature_len'],], name='term_feature')
    cluster_feature = Input(shape=[CONV_char_wide_params['cluster_feature_len'],], name='cluster_feature')
    # Embedding layers
    z = Embedding(input_dim=CONV_char_wide_params["vocab_size"] + 1, output_dim=CONV_char_wide_params['embedding_size'], input_length=CONV_char_wide_params['sequence_length'], trainable=True)(main_input)
    # Convolution layers
    convolution_output = []
    for num_filters, filter_width in CONV_char_wide_params["conv_layers"]:
        conv = Convolution1D(filters=num_filters, kernel_size=filter_width, activation='tanh', name='Conv1D_{}_{}'.format(num_filters, filter_width))(z)
        pool = GlobalMaxPooling1D(name='MaxPoolingOverTime_{}_{}'.format(num_filters, filter_width))(conv)
        convolution_output.append(pool)
    z = Concatenate()(convolution_output)
    # Fully connected layers
    for fc in CONV_char_wide_params["fc_layers"]:
        z = Dense(fc[0], activation='selu', kernel_initializer='lecun_normal')(z)
        z = AlphaDropout(fc[1])(z)
    pre_output = Concatenate()([z, term_feature, cluster_feature])
    model_output = Dense(2, activation="softmax")(pre_output)
    model = Model([main_input, term_feature, cluster_feature], model_output)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(decay=0.02, lr=0.0005), metrics=["mae", "categorical_accuracy"])
    # print(model.summary())
    return model

#================== Utils for word based cnn ==============

# read word2vec word embeddings, given desired type and vocab
def read_word2vec(name, embedding_type, vocab=None):
    if embedding_type == "gensim":
        model = gensim.models.Word2Vec.load(name)
        vectors = model.wv
    elif embedding_type == "google":
        vectors = ptvr.from_word2vec_binary(name, desired_vocab=vocab)
    elif embedding_type == "glove":
        vectors = ptvr.from_glove_plain(name, desired_vocab=vocab)
    elif embedding_type == "custom":
        with open(name, 'rb') as inf:
            vectors = pickle.load(inf)
    elif embedding_type == "godin" or embedding_type == "dsm":
        vectors = ptvr.from_word2vec_binary(name, encoding='utf8', newLines=True, desired_vocab=vocab)
    else:
        vectors = None
    return vectors

# build vocabulary base on a set of tweets
def build_vocab(tweets, params):
    # A better looking solution with Counter and itertools.chain
    word_counts = Counter(itertools.chain(*tweets))
    word_frequency_list = word_counts.most_common()
    print(word_frequency_list[0:5])
    # inv_vocab is now vocab_size+1, no need +1 later
    inv_vocab = ['__VOCABPLACEHOLDER__'] + [x[0] for x in word_frequency_list] # [0] position is a placeholder
    vocab = {x : i+1 for i,x in enumerate(inv_vocab[1:])} # reserve 0 for padding
    vocab['__VOCABPLACEHOLDER__'] = 0
    return vocab, inv_vocab, word_frequency_list

# build adj matrix for later determining the vector repr of unseen words
# def build_adj_matrix(tweets, vocab, inv_vocab):
    # init
    # vocab_size = len(inv_vocab)
    # two_left_adj_matrix = lil_matrix((vocab_size, vocab_size))
    # one_left_adj_matrix = lil_matrix((vocab_size, vocab_size))
    # one_right_adj_matrix = lil_matrix((vocab_size, vocab_size))
    # two_right_adj_matrix = lil_matrix((vocab_size, vocab_size))

    # scan all tweets and create adjency matrix
    # we have 4 adj matrix for context words in -2/-1/+1/+2 positions
    # for tweet in tweets:
    #     tweet_len = len(tweet)
    #     for i in range(len(tweet)):
    #         if i-2 >=0:
    #             two_left_adj_matrix[vocab[tweet[i]], vocab[tweet[i-2]]] += 1
    #         if i-1 >=0:
    #             one_left_adj_matrix[vocab[tweet[i]], vocab[tweet[i-1]]] += 1
    #         if i+1 >=0:
    #             one_right_adj_matrix[vocab[tweet[i]], vocab[tweet[i+1]]] += 1
    #         if i+2 >=0:
    #             two_right_adj_matrix[vocab[tweet[i]], vocab[tweet[i+2]]] += 1

    # return two_left_adj_matrix, one_left_adj_matrix, one_right_adj_matrix, two_right_adj_matrix
    # one_adj_matrix = {}
    # two_adj_matrix = {}
    # for comb in itertools.combinations(range(1, vocab_size), 2):
    #     one_adj_matrix[frozenset(comb)] = 0
    #     two_adj_matrix[frozenset(comb)] = 0

    # # build adj matrix 
    # for tweet in tweets:
    #     tweet_len = len(tweet)
    #     for i in range(len(tweet)):
    #         if i-2 >= 0:
    #             two_adj_matrix[frozenset(vocab[tweet[i]], vocab[tweet[i-2]])] += 1
    #         if i-1 >= 0:
    #             one_adj_matrix[frozenset(vocab[tweet[i]], vocab[tweet[i-1]])] += 1
    #         if i+1 < tweet_len:
    #             one_adj_matrix[frozenset(vocab[tweet[i]], vocab[tweet[i+1]])] += 1
    #         if i+2 < tweet_len:
    #             two_adj_matrix[frozenset(vocab[tweet[i]], vocab[tweet[i+2]])] += 1

    # return one_adj_matrix, two_adj_matrix

def save_pickle_file(filepath, obj):
    with open(filepath, 'wb') as outf:
        pickle.dump(obj, outf)

def load_pickle_file(filepath):
    with open(filepath, 'rb') as outf:
        obj = pickle.load(outf)
    return obj

def build_or_load_adj_matrix(tweets, vocab, inv_vocab, word_vecs, general_params):
    if os.path.isfile(general_params['temp_home'] + "two_left_adj_matrix.pik") and \
        os.path.isfile(general_params['temp_home'] + "one_left_adj_matrix.pik") and \
        os.path.isfile(general_params['temp_home'] + "one_right_adj_matrix.pik") and \
        os.path.isfile(general_params['temp_home'] + "two_right_adj_matrix.pik"):
        print("loading adj matrix")
        two_left_adj_matrix = load_pickle_file(general_params['temp_home'] + "two_left_adj_matrix.pik")
        one_left_adj_matrix = load_pickle_file(general_params['temp_home'] + "one_left_adj_matrix.pik")
        one_right_adj_matrix = load_pickle_file(general_params['temp_home'] + "one_right_adj_matrix.pik")
        two_right_adj_matrix = load_pickle_file(general_params['temp_home'] + "two_right_adj_matrix.pik")
    else:
        print("building adj matrix")
        # init
        vocab_size = len(inv_vocab)
        two_left_adj_matrix = lil_matrix((vocab_size, vocab_size))
        one_left_adj_matrix = lil_matrix((vocab_size, vocab_size))
        one_right_adj_matrix = lil_matrix((vocab_size, vocab_size))
        two_right_adj_matrix = lil_matrix((vocab_size, vocab_size))

        # scan all tweets and create adjency matrix
        # we have 4 adj matrix for context words in -2/-1/+1/+2 positions
        count = 0
        for tweet in tweets:
            count += 1
            if count % 10000 == 0:
                print('processed 10000 tweets for adj matrix')
            tweet_len = len(tweet)
            for i in range(len(tweet)):
                if i-2 >=0:
                    two_left_adj_matrix[vocab[tweet[i]], vocab[tweet[i-2]]] += 1
                if i-1 >=0:
                    one_left_adj_matrix[vocab[tweet[i]], vocab[tweet[i-1]]] += 1
                if i+1 < tweet_len:
                    one_right_adj_matrix[vocab[tweet[i]], vocab[tweet[i+1]]] += 1
                if i+2 < tweet_len:
                    two_right_adj_matrix[vocab[tweet[i]], vocab[tweet[i+2]]] += 1
        save_pickle_file(general_params['temp_home'] + "two_left_adj_matrix.pik", two_left_adj_matrix)
        save_pickle_file(general_params['temp_home'] + "one_left_adj_matrix.pik", two_left_adj_matrix)
        save_pickle_file(general_params['temp_home'] + "one_right_adj_matrix.pik", two_left_adj_matrix)
        save_pickle_file(general_params['temp_home'] + "two_right_adj_matrix.pik", two_left_adj_matrix)
    return two_left_adj_matrix, one_left_adj_matrix, one_right_adj_matrix, two_right_adj_matrix


# build word matrix, now with adj matrix included
def build_word_matrix(tweets, vocab, inv_vocab, word_vecs, word_frequency_list, wordvec_params, general_params):
    """
    Get word matrix. word_matrix[i] is the vector for word indexed by i
    If a word does not have vector repr in word_vecs, then use the average of its adj words as its repr
    If this word's adj words are also unseen, then use the average of 1% infrequent words as its repr
    """
    # init
    vocab_size = len(inv_vocab)
    
    # build or load adj matrix
    two_left_adj_matrix, one_left_adj_matrix, one_right_adj_matrix, two_right_adj_matrix = build_or_load_adj_matrix(tweets, vocab, inv_vocab, word_vecs, general_params)
    # two_left_adj_matrix_csr = two_left_adj_matrix.tocsr()
    # one_left_adj_matrix_csr = one_left_adj_matrix.tocsr()
    # one_right_adj_matrix_csr = one_right_adj_matrix.tocsr()
    # two_right_adj_matrix_csr = two_right_adj_matrix.tocsr()

    # pre-sort the adj matrix
    # two_left_adj_counts = [two_left_adj_matrix[i].toarray().flatten() for i in range(vocab_size)]
    # one_left_adj_counts = [one_left_adj_matrix[i].toarray().flatten() for i in range(vocab_size)]
    # one_right_adj_counts = [one_right_adj_matrix[i].toarray().flatten() for i in range(vocab_size)]
    # two_right_adj_counts = [two_right_adj_matrix[i].toarray().flatten() for i in range(vocab_size)]

    # scan all tweets and create adjency matrix
    # we have 4 adj matrix for context words in -2/-1/+1/+2 positions
    # count = 0
    # for tweet in tweets:
    #     count += 1
    #     if count % 10000 == 0:
    #         print('processed 10000 tweets for adj matrix')
    #     tweet_len = len(tweet)
    #     for i in range(len(tweet)):
    #         if i-2 >=0:
    #             two_left_adj_matrix[vocab[tweet[i]], vocab[tweet[i-2]]] += 1
    #         if i-1 >=0:
    #             one_left_adj_matrix[vocab[tweet[i]], vocab[tweet[i-1]]] += 1
    #         if i+1 < tweet_len:
    #             one_right_adj_matrix[vocab[tweet[i]], vocab[tweet[i+1]]] += 1
    #         if i+2 < tweet_len:
    #             two_right_adj_matrix[vocab[tweet[i]], vocab[tweet[i+2]]] += 1
    
    # determine infrequent word vec
    infreq_word_size = round(vocab_size / 100.0 * 1.0)
    infreq_word_freq = word_frequency_list[-infreq_word_size][1]
    infreq_word_vec = np.zeros(shape=(1, wordvec_params['embedding_dim']))
    i = 0
    for word, count in word_frequency_list[-infreq_word_size:]:
        if word in word_vecs:
            i += 1
            infreq_word_vec += word_vecs[word]
    infreq_word_vec = infreq_word_vec / i
    #print(infreq_word_vec)

    # build word matrix
    word_matrix = np.zeros(shape=(vocab_size, wordvec_params['embedding_dim']))
    infreq_word_count = 0
    infreq_words = []
    for i in range(1, vocab_size):
        # word in word vecs
        if inv_vocab[i] in word_vecs:
            word_matrix[i] = word_vecs[inv_vocab[i]] # 0 will be used as padding, so start from 1
        # use something to represent the unknown word
        else:
            # try use capitalized version of the word
            cap_word = inv_vocab[i][0].upper() + inv_vocab[i][1:]
            allcap_word = inv_vocab[i].upper()
            if cap_word in word_vecs:
                word_matrix[i] = word_vecs[cap_word]
            elif allcap_word in word_vecs:
                word_matrix[i] = word_vecs[allcap_word]
            else:
                # if this word is very infrequent, use averaged infreq_vec
                if word_frequency_list[vocab[word] - 1][1] <= infreq_word_freq:
                    word_matrix[i] = infreq_word_vec
                    continue
                infreq_words.append(inv_vocab[i])
                infreq_word_count += 1
                if infreq_word_count % 10000 == 0:
                    print("found 10000 infreq words")
                # fetch the row of adj counts of this word in each adj matrix
                two_left_adj_count = two_left_adj_matrix[i].toarray().flatten()
                one_left_adj_count = one_left_adj_matrix[i].toarray().flatten()
                one_right_adj_count = one_right_adj_matrix[i].toarray().flatten()
                two_right_adj_count = two_right_adj_matrix[i].toarray().flatten()
                # for each position find the most occured word and get the word_vecs
                # if the word's context are also unseen words, use infrequent words as the repr
                context_avail = []
                context_vecs = np.zeros(shape=(1, wordvec_params['embedding_dim']))
                context_index = []
                context_index.append(np.argsort(two_left_adj_count)[-1])
                if two_left_adj_count[context_index[0]] > 0 and inv_vocab[context_index[0]] in word_vecs:
                    context_avail.append(1)
                else:
                    context_avail.append(0)
                context_index.append(np.argsort(one_left_adj_count)[-1])
                if one_left_adj_count[context_index[1]] > 0 and inv_vocab[context_index[1]] in word_vecs:
                    context_avail.append(1)
                else:
                    context_avail.append(0)
                context_index.append(np.argsort(one_right_adj_count)[-1])
                if one_right_adj_count[context_index[2]] > 0 and inv_vocab[context_index[2]] in word_vecs:
                    context_avail.append(1)
                else:
                    context_avail.append(0)
                context_index.append(np.argsort(two_right_adj_count)[-1])
                if two_right_adj_count[context_index[3]] > 0 and inv_vocab[context_index[3]] in word_vecs:
                    context_avail.append(1)
                else:
                    context_avail.append(0)
                # enough context words
                if sum(context_avail) >= 2:
                    for i in range(4):
                        if context_avail[i] > 0:
                            # sum existing word vecs
                            context_vecs += word_vecs[inv_vocab[context_index[i]]]
                    # average the word vecs as this infrequent word's vec
                    word_matrix[i] = context_vecs / sum(context_avail)
                # not enough context words, use averaged infrequent words
                else:
                    word_matrix[i] = infreq_word_vec
            
            # generate frozenset of all word pairs for accessing the adj matrix
            #adj_keys = [frozenset(x) for x in list(itertools.product([i], list(range(1, i) + list(range(i+1, vocab_size)))))]
            # access the count of each word adjcent to target word
            #one_adj_counts = [one_adj_matrix[key] for key in adj_keys]
            # sort the count for finding the highest counts
            #one_adj_sorted_idx = np.argsort(one_adj_counts)
            #two_adj_counts = [two_adj_matrix[key] for key in adj_keys]
            #two_adj_sorted_idx = np.argsort(two_adj_counts)
    with open(general_params['temp_home'] + 'infreq_words.txt', 'w') as outf:
        for word in infreq_words:
            outf.write(word + '\n')

    # save word matrix to file
    with open(general_params['temp_home'] + wordvec_params['wordmatrix_file'], 'wb') as outf:
        pickle.dump(word_matrix, outf)
    return word_matrix

def build_word_matrix2(tweets, vocab, inv_vocab, word_vecs, word_frequency_list, wordvec_params, general_params):
    """
    Get word matrix. word_matrix[i] is the vector for word indexed by i
    If a word does not have vector repr in word_vecs, then use the average of its adj words as its repr
    If this word's adj words are also unseen, then use the average of 1% infrequent words as its repr
    """
    # init
    vocab_size = len(inv_vocab)
    
    # build or load adj matrix
    two_left_adj_matrix, one_left_adj_matrix, one_right_adj_matrix, two_right_adj_matrix = build_or_load_adj_matrix(tweets, vocab, inv_vocab, word_vecs, general_params)
    
    # determine infrequent word vec
    infreq_word_size = 2000
    infreq_word_freq = 2
    infreq_word_vec = np.zeros(shape=(1, wordvec_params['embedding_dim']))
    i = 0
    for word, count in word_frequency_list[-infreq_word_size:]:
        if word in word_vecs:
            i += 1
            infreq_word_vec += word_vecs[word]
    infreq_word_vec = infreq_word_vec / i
    #print(infreq_word_vec)

    # build word matrix
    word_matrix = np.zeros(shape=(vocab_size, wordvec_params['embedding_dim']))
    infreq_word_count = 0
    infreq_words = []
    for i in range(1, vocab_size):
        # word in word vecs
        if inv_vocab[i] in word_vecs:
            word_matrix[i] = word_vecs[inv_vocab[i]] # 0 will be used as padding, so start from 1
        # use something to represent the unknown word
        else:
            # try use capitalized version of the word
            cap_word = inv_vocab[i][0].upper() + inv_vocab[i][1:]
            allcap_word = inv_vocab[i].upper()
            if cap_word in word_vecs:
                word_matrix[i] = word_vecs[cap_word]
            elif allcap_word in word_vecs:
                word_matrix[i] = word_vecs[allcap_word]
            else:
                # if this word is very infrequent, use averaged infreq_vec
                if word_frequency_list[vocab[word] - 1][1] < infreq_word_freq:
                    word_matrix[i] = infreq_word_vec
                    continue
                # if this word is somewhat frequent
                infreq_words.append(inv_vocab[i])
                infreq_word_count += 1
                if infreq_word_count % 10000 == 0:
                    print("found 10000 infreq words")
                # fetch the row of adj counts of this word in each adj matrix
                two_left_adj_count = two_left_adj_matrix[i].toarray().flatten()
                one_left_adj_count = one_left_adj_matrix[i].toarray().flatten()
                one_right_adj_count = one_right_adj_matrix[i].toarray().flatten()
                two_right_adj_count = two_right_adj_matrix[i].toarray().flatten()
                # for each position find the most occured word and get the word_vecs
                # if the word's context are also unseen words, use infrequent words as the repr
                context_avail = []
                context_vecs = np.zeros(shape=(1, wordvec_params['embedding_dim']))
                context_index = []
                context_index.append(np.argsort(two_left_adj_count)[-1])
                if two_left_adj_count[context_index[0]] > 0 and inv_vocab[context_index[0]] in word_vecs:
                    context_avail.append(1)
                else:
                    context_avail.append(0)
                context_index.append(np.argsort(one_left_adj_count)[-1])
                if one_left_adj_count[context_index[1]] > 0 and inv_vocab[context_index[1]] in word_vecs:
                    context_avail.append(1)
                else:
                    context_avail.append(0)
                context_index.append(np.argsort(one_right_adj_count)[-1])
                if one_right_adj_count[context_index[2]] > 0 and inv_vocab[context_index[2]] in word_vecs:
                    context_avail.append(1)
                else:
                    context_avail.append(0)
                context_index.append(np.argsort(two_right_adj_count)[-1])
                if two_right_adj_count[context_index[3]] > 0 and inv_vocab[context_index[3]] in word_vecs:
                    context_avail.append(1)
                else:
                    context_avail.append(0)
                # enough context words
                if sum(context_avail) >= 2:
                    for i in range(4):
                        if context_avail[i] > 0:
                            # sum existing word vecs
                            context_vecs += word_vecs[inv_vocab[context_index[i]]]
                    # average the word vecs as this infrequent word's vec
                    word_matrix[i] = context_vecs / sum(context_avail)
                # not enough context words, use averaged infrequent words
                else:
                    word_matrix[i] = infreq_word_vec
            
    with open(general_params['temp_home'] + 'infreq_words.txt', 'w') as outf:
        for word in infreq_words:
            outf.write(word + '\n')

    # save word matrix to file
    with open(general_params['temp_home'] + wordvec_params['wordmatrix_file_syn_expan'], 'wb') as outf:
        pickle.dump(word_matrix, outf)
    return word_matrix

def build_word_matrix3(tweets, vocab, inv_vocab, word_vecs, word_frequency_list, wordvec_params, general_params, wordmatrix_file, no_save):
    """
    Get word matrix. word_matrix[i] is the vector for word indexed by i
    If a word does not have vector repr in word_vecs, then use the average of its adj words as its repr
    If this word's adj words are also unseen, then use the average of 1% infrequent words as its repr
    """
    # init
    vocab_size = len(inv_vocab)
    
    # build or load adj matrix
    two_left_adj_matrix, one_left_adj_matrix, one_right_adj_matrix, two_right_adj_matrix = build_or_load_adj_matrix(tweets, vocab, inv_vocab, word_vecs, general_params)
    
    # determine infrequent word vec
    infreq_word_size = 2000
    infreq_word_freq = 2
    infreq_word_vec = np.zeros(shape=(1, wordvec_params['embedding_dim']))
    i = 0
    for word, count in word_frequency_list[-infreq_word_size:]:
        if word in word_vecs:
            i += 1
            infreq_word_vec += word_vecs[word]
    infreq_word_vec = infreq_word_vec / i
    # print(infreq_word_vec)

    # build word matrix
    word_matrix = np.zeros(shape=(vocab_size, wordvec_params['embedding_dim']))
    infreq_word_count = 0
    infreq_words = []
    for i in range(1, vocab_size):
        # word in word vecs
        if inv_vocab[i] in word_vecs:
            word_matrix[i] = word_vecs[inv_vocab[i]] # 0 will be used as padding, so start from 1
        # use something to represent the unknown word
        else:
            # try use capitalized version of the word
            cap_word = inv_vocab[i][0].upper() + inv_vocab[i][1:]
            allcap_word = inv_vocab[i].upper()
            if cap_word in word_vecs:
                word_matrix[i] = word_vecs[cap_word]
            elif allcap_word in word_vecs:
                word_matrix[i] = word_vecs[allcap_word]
            else:
                # if this word is very infrequent, use averaged infreq_vec
                if word_frequency_list[vocab[word] - 1][1] < infreq_word_freq:
                    word_matrix[i] = infreq_word_vec
                    continue
                # if this word is somewhat frequent
                infreq_words.append(inv_vocab[i])
                infreq_word_count += 1
                if infreq_word_count % 10000 == 0:
                    print("found 10000 infreq words")
                # fetch the row of adj counts of this word in each adj matrix
                two_left_adj_count = two_left_adj_matrix[i].toarray().flatten()
                one_left_adj_count = one_left_adj_matrix[i].toarray().flatten()
                one_right_adj_count = one_right_adj_matrix[i].toarray().flatten()
                two_right_adj_count = two_right_adj_matrix[i].toarray().flatten()
                # for each position find the most occured word and get the word_vecs
                # if the word's context are also unseen words, use infrequent words as the repr
                context_avail = []
                context_vecs = np.zeros(shape=(1, wordvec_params['embedding_dim']))
                context_index = []
                context_index.append(np.argsort(two_left_adj_count)[-1])
                if two_left_adj_count[context_index[0]] > 0 and inv_vocab[context_index[0]] in word_vecs:
                    context_avail.append(1)
                else:
                    context_avail.append(0)
                context_index.append(np.argsort(one_left_adj_count)[-1])
                if one_left_adj_count[context_index[1]] > 0 and inv_vocab[context_index[1]] in word_vecs:
                    context_avail.append(1)
                else:
                    context_avail.append(0)
                context_index.append(np.argsort(one_right_adj_count)[-1])
                if one_right_adj_count[context_index[2]] > 0 and inv_vocab[context_index[2]] in word_vecs:
                    context_avail.append(1)
                else:
                    context_avail.append(0)
                context_index.append(np.argsort(two_right_adj_count)[-1])
                if two_right_adj_count[context_index[3]] > 0 and inv_vocab[context_index[3]] in word_vecs:
                    context_avail.append(1)
                else:
                    context_avail.append(0)
                # enough context words
                if sum(context_avail) >= 2:
                    for i in range(4):
                        if context_avail[i] > 0:
                            # sum existing word vecs
                            context_vecs += word_vecs[inv_vocab[context_index[i]]]
                    # average the word vecs as this infrequent word's vec
                    word_matrix[i] = context_vecs / sum(context_avail)
                # not enough context words, use averaged infrequent words
                else:
                    word_matrix[i] = infreq_word_vec
            
    with open(general_params['temp_home'] + 'infreq_words.txt', 'w') as outf:
        for word in infreq_words:
            outf.write(word + '\n')

    # save word matrix to file
    if not no_save:
        with open(general_params['temp_home'] + wordmatrix_file, 'wb') as outf:
            pickle.dump(word_matrix, outf)
    return word_matrix

def build_vocab_and_matrix(wordmatrix_file, all_tweets, general_params, wordvec_params, no_save=False):
    # load everythin for LSTM/CONV
    print('building vocab')
    vocab, inv_vocab, word_frequency_list = build_vocab(all_tweets, wordvec_params)

    if no_save:
        print('reading word2vec')
        word_vecs = read_word2vec(general_params['wordvec_home'] + wordvec_params["word2vec_file"], wordvec_params["embedding"], vocab)
        print('building word matrix')
        word_matrix = build_word_matrix3(all_tweets, vocab, inv_vocab, word_vecs, word_frequency_list, wordvec_params, general_params, wordmatrix_file, no_save=True)
        print("vocab size = {}".format(word_matrix.shape[0]))
    else:
        print(general_params['temp_home'] + wordmatrix_file)
        # check existence of word matrix file
        word_matrix = None
        word_matrix_flag = False
        if os.path.isfile(general_params['temp_home'] + wordmatrix_file):
            with open(general_params['temp_home'] + wordmatrix_file, 'rb') as inf:
                word_matrix = pickle.load(inf)
            # correct word matrix
            print(word_matrix.shape)
            print(len(inv_vocab))
            print(wordvec_params['embedding_dim'])
            if word_matrix.shape[0] == len(inv_vocab) and word_matrix.shape[1] == wordvec_params['embedding_dim']:
                word_matrix_flag = True
                print('word matrix found')
        
        if not word_matrix_flag:
            print('reading word2vec')
            word_vecs = read_word2vec(general_params['wordvec_home'] + wordvec_params["word2vec_file"], wordvec_params["embedding"], vocab)
            print('building word matrix')
            word_matrix = build_word_matrix3(all_tweets, vocab, inv_vocab, word_vecs, word_frequency_list, wordvec_params, general_params, wordmatrix_file, no_save=False)
            print("vocab size = {}".format(word_matrix.shape[0]))
            # free some memory
            del word_vecs
    return vocab, inv_vocab, word_frequency_list, word_matrix


# transform tweets (list of words) to ndarray of vectors
def transform_word_seq_with_padding(tweets, vocab, max_len=20, direction='pre'):
    # Better looking solution with one line of code
    tweets_word_index = [[vocab[w] for w in t if w in vocab] for t in tweets]
    return pad_sequences(tweets_word_index, maxlen=max_len, padding=direction, truncating=direction)

def transform_word_seq_with_padding_masking(tweets, mask_words, vocab, max_len=20, direction='pre'):
    tweets_word_index = []
    for t in tweets:
        t_w = []
        for word in t:
            if word in mask_words:
                t_w.append(0)
            elif word in vocab:
                t_w.append(vocab[word])
        tweets_word_index.append(t_w)
    #tweets_word_index = [[vocab[w] for w in t if w in vocab] for t in tweets]
    return pad_sequences(tweets_word_index, maxlen=max_len, padding=direction, truncating=direction)

# ================== Utils for char based cnn ==================================

def transform_char_seq_with_padding(tweets, max_len=280, direction='post'):
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n"
    char_dict = {}
    for i in range(len(alphabet)):
        char_dict[alphabet[i]] = i + 1
    index_seq_list = []
    for t in tweets:
        index_array = np.zeros(max_len, dtype='int64')
        if type(t) != str: 
            string = (' '.join(t)).lower()
            for i in range(len(string)):
                if i >= max_len:
                    break
                if string[i] in char_dict:
                    index_array[i] = char_dict[string[i]]
            #index_seq_list.append(np.asarray([char_dict[c] if c in alphabet else 0 for c in ' '.join(t).lower()]))
        else:
            string = t.lower()
            for i in range(len(string)):
                if i >= max_len:
                    break
                if string[i] in char_dict:
                    index_array[i] = char_dict[string[i]]
            #index_seq_list.append(np.asarray([char_dict[c] if c in alphabet else 0 for c in t.lower()], dtype='int64'))
        index_seq_list.append(index_array)
    return np.asarray(index_seq_list)
    # return pad_sequences(index_seq_list, maxlen=max_len, padding=direction, truncating=direction)

# ======================= train, save and load models ===========================
def if_train_load_LSTM(train_vectors, train_labels, word_matrix, LSTM_model_file, LSTM_params, wordvec_params, force_train=False):
    if os.path.isfile(LSTM_model_file) and not force_train:
        print("LSTM model found, load it")
        return load_model(LSTM_model_file)
    else:
        print("No LSTM model found, train a new one and save")
        model = model_lstm_relu_embedding(word_matrix, LSTM_params, wordvec_params)
        train_labels_onehot = to_categorical(train_labels)
        model.fit(train_vectors, train_labels_onehot, epochs=LSTM_params["lstm_epochs"],
            batch_size=LSTM_params["batch_size"],callbacks=None,verbose=LSTM_params['verbose'])
        model.save(LSTM_model_file)
        return model

def train_LSTM(train_vectors, train_labels, word_matrix, LSTM_params, wordvec_params):
    print("Train LSTM model without saving")
    model = model_lstm_relu_embedding(word_matrix, LSTM_params, wordvec_params)
    train_labels_onehot = to_categorical(train_labels)
    model.fit(train_vectors, train_labels_onehot, epochs=LSTM_params["lstm_epochs"],
        batch_size=LSTM_params["batch_size"],callbacks=None,verbose=LSTM_params['verbose'])
    return model

def load_LSTM(LSTM_model_file):
    if os.path.isfile(LSTM_model_file):
        print("LSTM model {} found, load it".format(LSTM_model_file))
        return load_model(LSTM_model_file)
    else:
        print("LSTM model {} not found".format(LSTM_model_file))
        return None


def if_train_load_CONV(train_vectors, train_labels, word_matrix, CONV_model_file, CONV_params, wordvec_params, force_train=False):
    if os.path.isfile(CONV_model_file) and not force_train:
        print("CONV model found, load it")
        return load_model(CONV_model_file)
    else:
        print("No CONV model found, train a new one and save")
        model = model_conv2d_relu_embedding(word_matrix, CONV_params, wordvec_params)
        train_labels_onehot = to_categorical(train_labels)
        model.fit(train_vectors, train_labels_onehot, epochs=CONV_params["conv_epochs"],
            batch_size=CONV_params["batch_size"],callbacks=None,verbose=CONV_params['verbose'])
        model.save(CONV_model_file)
        return model

# def train_CONV(model, train_data, test_data, general_params, CONV_params):
#     print("Train CONV model without saving")
#     call_back = LogEpochPerformance({'main_input':test_vectors}, test_labels, general_params['epoch_log_filename'], general_params['every_n_epoch'])
#     model.fit(train_vectors, train_labels_onehot, epochs=CONV_params["conv_epochs"],
#         batch_size=CONV_params["batch_size"],callbacks=[call_back],verbose=CONV_params['verbose'])
#     return model

# def train_CONV_class_weighted(model, train_vectors, train_labels, test_vectors, test_labels, general_params, CONV_params):
#     print("Train CONV model without saving")
#     train_labels_onehot = to_categorical(train_labels)
#     call_back = LogEpochPerformance({'main_input':test_vectors}, test_labels, general_params['epoch_log_filename'], general_params['every_n_epoch'])
#     model.fit(train_vectors, train_labels_onehot, epochs=CONV_params["conv_epochs"],
#         batch_size=CONV_params["batch_size"],callbacks=[call_back],verbose=CONV_params['verbose'], 
#         class_weight=CONV_params['class_weight'])
#     return model

# def train_CONV_extra_feature_concat_class_weighted(model, train_vectors, train_labels, term_features, clu_features, test_vectors, test_labels, general_params, CONV_params):
#     print("Train CONV model without saving")
#     train_labels_onehot = to_categorical(train_labels)
#     call_back = LogEpochPerformance({'main_input':test_vectors, 'term_feature':term_features, 'cluster_feature':clu_features}, test_labels, general_params['epoch_log_filename'], general_params['every_n_epoch'])
#     model.fit({'main_input':train_vectors, 'term_feature':term_features, 'cluster_feature':clu_features}, train_labels_onehot, epochs=CONV_params["conv_epochs"],
#         batch_size=CONV_params["batch_size"],callbacks=None,verbose=CONV_params['verbose'], 
#         class_weight=CONV_params['class_weight'])
#     return model

# def train_CONV_extra_feature_parallel_class_weighted(model, train_vectors, train_labels, syn_expan, term_features, clu_features, CONV_params, wordvec_params):
#     print("Train CONV model without saving")
#     train_labels_onehot = to_categorical(train_labels)
#     model.fit({'main_input':train_vectors, 'syn_expan':syn_expan, 'term_feature':term_features, 'cluster_feature':clu_features}, train_labels_onehot, epochs=CONV_params["conv_epochs"],
#         batch_size=CONV_params["batch_size"],callbacks=None,verbose=CONV_params['verbose'], 
#         class_weight=CONV_params['class_weight'])
#     return model

def train_char_CONV(model, train_vectors, train_labels, CONV_params):
    print("Train CONV model without saving")
    train_labels_onehot = to_categorical(train_labels)
    model.fit(train_vectors, train_labels_onehot, epochs=CONV_params["conv_epochs"],
        batch_size=CONV_params["batch_size"],callbacks=None,verbose=CONV_params['verbose'])
    return model

def train_char_CONV_class_weighted(model, train_vectors, train_labels, CONV_params):
    print("Train CONV model without saving")
    train_labels_onehot = to_categorical(train_labels)
    model.fit(train_vectors, train_labels_onehot, epochs=CONV_params["conv_epochs"],
        batch_size=CONV_params["batch_size"],callbacks=None,verbose=CONV_params['verbose'], 
        class_weight=CONV_params['class_weight'])
    return model


def load_CONV(CONV_model_file):
    if os.path.isfile(CONV_model_file):
        print("CONV model {} found, load it".format(CONV_model_file))
        return load_model(CONV_model_file)
    else:
        print("CONV model {} not found".format(CONV_model_file))
        return None
