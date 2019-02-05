# set to use specific GPU
# set to '' to use cpu
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# set to not allocate all gpu memory
# but allow memory growth
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from keras.backend.tensorflow_backend import set_session
set_session(tf.Session(config=config))

from keras import backend as K

from keras.backend.tensorflow_backend import clear_session
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
from keras.models import load_model
from keras.utils import to_categorical
from collections import Counter
import datetime

import pandas as pd
import numpy as np
import sys,pickle
from argparse import ArgumentParser

from LSTM_CONV import *

from training_params import *

from training_utils import *


# read expanded features for word-based cnn models
def prepare_feature_expan_data(general_params, CONV_params, wordvec_params):
    # read tweets
    raw_tweets, clean_tweets, labels = read_labeled_data(general_params, shuffle=False)

    # read additional features
    hl_syn_expan_lists = read_pickle(general_params['temp_home'] + general_params['hl_syn_expan_lists_filename'])
    hl_term_features = read_pickle(general_params['temp_home'] + general_params['hl_term_feature_filename']).toarray()
    hl_cluster_features = read_pickle(general_params['temp_home'] + general_params['hl_cluster_feature_filename']).toarray()
    print(clean_tweets.shape)
    print('-----------------')
    
    # concat hl syn_expan to tweets
    hl_expanded_tweets = []
    for i in range(clean_tweets.shape[0]):
        c = len(hl_syn_expan_lists[i])
        if len(clean_tweets[i]) + len(hl_syn_expan_lists[i]) > CONV_params['sequence_length']:
            c = CONV_params['sequence_length'] - len(clean_tweets[i])
        combined_words = hl_syn_expan_lists[i][0:c] + clean_tweets[i]
        hl_expanded_tweets.append(combined_words)
    hl_expanded_tweets = np.asarray(hl_expanded_tweets)

    # concatenate all tweets for making vocabulary
    print("getting word matrix")
    vocab, inv_vocab, word_frequency_list, word_matrix = build_vocab_and_matrix(wordvec_params['wordmatrix_file_syn_expan2'], hl_expanded_tweets, general_params, wordvec_params)

    return clean_tweets, hl_expanded_tweets, labels, hl_term_features, hl_cluster_features, vocab, word_matrix

# read expanded features for char-based cnn
def prepare_feature_expan_data_no_matrix(general_params, CONV_params, wordvec_params):
    # read tweets
    raw_tweets, clean_tweets, labels = read_labeled_data(general_params, shuffle=False)

    # read additional features
    hl_syn_expan_lists = read_pickle(general_params['temp_home'] + general_params['hl_syn_expan_lists_filename'])
    hl_term_features = read_pickle(general_params['temp_home'] + general_params['hl_term_feature_filename']).toarray()
    hl_cluster_features = read_pickle(general_params['temp_home'] + general_params['hl_cluster_feature_filename']).toarray()
    print(clean_tweets.shape)
    print('-----------------')

    # concat hl syn_expan to tweets
    hl_expanded_tweets = []
    for i in range(clean_tweets.shape[0]):
        c = len(hl_syn_expan_lists[i])
        if len(clean_tweets[i]) + len(hl_syn_expan_lists[i]) > CONV_params['sequence_length']:
            c = CONV_params['sequence_length'] - len(clean_tweets[i])
        combined_words = hl_syn_expan_lists[i][0:c] + clean_tweets[i]
        hl_expanded_tweets.append(combined_words)
    hl_expanded_tweets = np.asarray(hl_expanded_tweets)

    return clean_tweets, hl_expanded_tweets, labels, hl_term_features, hl_cluster_features

# return model given string name
def model_select(model_type, CONV_params, word_matrix, wordvec_params):
    if model_type == 'char_wide':
        base_model = model_wide_cnn_char(CONV_params)
    elif model_type == 'char_deep':
        base_model = model_deep_cnn_char(CONV_params)
    elif model_type == 'char_expan':
        base_model = model_wide_cnn_char_extra_feature(CONV_params)
    elif model_type == 'word_wide':
        base_model = model_wide_cnn_word(word_matrix, CONV_params, wordvec_params)
    elif model_type == 'word_deep':
        base_model = model_deep_cnn_word(word_matrix, CONV_params, wordvec_params)
    elif model_type == 'word_both':
        base_model = model_wide_deep_cnn_word(word_matrix, CONV_params, wordvec_params)
    elif model_type == 'word_expan_concat':
        base_model = model_wide_deep_cnn_word_extra_feature_concat(word_matrix, CONV_params, wordvec_params)
    elif model_type == 'word_expan_para':
        base_model = model_wide_deep_cnn_word_extra_feature_parallel(word_matrix, CONV_params, wordvec_params)
    else:
        print('model type invalid')
        exit()
    return base_model

# get config given string name
def model_param_select(model_type):
    if model_type == 'char_wide':
        params = get_CONV_char_wide_params()
    elif model_type == 'char_deep':
        params = get_CONV_char_deep_params()
    elif model_type == 'char_expan':
        params = get_CONV_char_wide_params()
    elif model_type == 'word_wide':
        params = get_CONV_wide_params()
    elif model_type == 'word_deep':
        params = get_CONV_deep_params()
    elif model_type == 'word_both':
        params = get_CONV_wide_deep_params()
    elif model_type == 'word_expan_concat':
        params = get_CONV_wide_deep_params()
    elif model_type == 'word_expan_para':
        params = get_CONV_wide_deep_syn_expan_params()
    else:
        print('model type invalid')
        exit()
    return params

# ensemble same types of models
# log each model's performance when training
# save all results to file and ensemble later
def paralle_ensemble(general_params, cw, index_number, model_types, test):
    # read data
    raw_tweets, clean_tweets, labels = read_labeled_data(general_params, shuffle=False)

    # read index
    kfold_index_lists = get_index(raw_tweets, labels, general_params, index_number, force_new=False)

    # prepare dict of results for ensemble
    testing_labels = []
    testing_preds = []
    fold_valid_labels = {}
    fold_valid_preds = {}
    all_fold_results = {}

    # prepare folder to save results
    results_save_path = general_params['results_home'] + 'ensemble_cnn_{}_p_{}_n_{}_index_{}/'.format('_'.join(model_types),
                general_params['label_ratio'][1], general_params['label_ratio'][0], index_number)
    if not os.path.exists(results_save_path):
        os.mkdir(results_save_path)

    # train model one by one
    # loop fold
    fold = 0
    for index_dict in kfold_index_lists:
        # prepare filename for each fold
        epoch_result_filename = results_save_path + "fold_{}_epoch_log.pik".format(fold)
        test_result_filename = results_save_path + "testing_results.pik"

        # train each model
        # loop model
        fold_valid_labels[fold] = []
        fold_valid_preds[fold] = []
        model_count = 0
        for mt in model_types:
            model_count += 1
            print('fold {}, model {}'.format(fold, model_count))
            epoch_test_labels = []
            epoch_test_preds = []
            epoch_valid_labels = []
            epoch_valid_preds = []
            CONV_params = model_param_select(mt)
            CONV_params['class_weight'] = cw
            wordvec_params = get_DSM_wordvec_params()
            
            if mt == 'word_expan_concat':
                # split train test data
                clean_tweets, hl_expanded_tweets, labels, hl_term_features, hl_cluster_features, vocab, word_matrix = prepare_feature_expan_data(general_params, CONV_params, wordvec_params)

                test_tweets = hl_expanded_tweets[index_dict['test']]
                test_labels = labels[index_dict['test']]
                test_labels_onehot = to_categorical(test_labels)
                test_term_features = hl_term_features[index_dict['test']]
                test_cluster_features = hl_cluster_features[index_dict['test']]
                test_word_vectors = transform_word_seq_with_padding(test_tweets, vocab, max_len=CONV_params["sequence_length"], direction='pre')
                print("test_word_vectors size {}, test_label size {}".format(test_word_vectors.shape, test_labels.shape))
                
                if test and fold == 0:
                    final_train_tweets = hl_expanded_tweets[index_dict['valid'] + index_dict['train']]
                    final_train_labels = labels[index_dict['valid'] + index_dict['train']]
                    final_train_labels_onehot = to_categorical(final_train_labels)
                    final_train_term_features = hl_term_features[index_dict['valid'] + index_dict['train']]
                    final_train_cluster_features = hl_cluster_features[index_dict['valid'] + index_dict['train']]
                    
                    # vectorize tweets
                    final_train_word_vectors = transform_word_seq_with_padding(final_train_tweets, vocab, max_len=CONV_params["sequence_length"], direction='pre')
                    print("final_train_word_vectors size {}, final_train_label size {}".format(final_train_word_vectors.shape, final_train_labels.shape))
                    test_model = model_select(mt, CONV_params, word_matrix, wordvec_params)
                else:
                    train_tweets = hl_expanded_tweets[index_dict['train']]
                    train_labels = labels[index_dict['train']]
                    train_labels_onehot = to_categorical(train_labels)
                    train_term_features = hl_term_features[index_dict['train']]
                    train_cluster_features = hl_cluster_features[index_dict['train']]

                    valid_tweets = hl_expanded_tweets[index_dict['valid']]
                    valid_labels = labels[index_dict['valid']]
                    valid_labels_onehot = to_categorical(valid_labels)
                    valid_term_features = hl_term_features[index_dict['valid']]
                    valid_cluster_features = hl_cluster_features[index_dict['valid']]

                    # vectorize tweets
                    train_word_vectors = transform_word_seq_with_padding(train_tweets, vocab, max_len=CONV_params["sequence_length"], direction='pre')
                    valid_word_vectors = transform_word_seq_with_padding(valid_tweets, vocab, max_len=CONV_params["sequence_length"], direction='pre')
                    print("train_word_vectors size {}, train_label size {}".format(train_word_vectors.shape, train_labels.shape))
                    print("valid_word_vectors size {}, valid_label size {}".format(valid_word_vectors.shape, valid_labels.shape))
                    
                    base_model = model_select(mt, CONV_params, word_matrix, wordvec_params)
                    

            elif mt == 'char_wide':
                test_tweets = raw_tweets[index_dict['test']]
                test_labels = labels[index_dict['test']]
                test_labels_onehot = to_categorical(test_labels)
                test_char_vectors = transform_char_seq_with_padding(test_tweets, max_len=280, direction='post')
                print("test_char_vectors size {}, test_label size {}".format(test_char_vectors.shape, test_labels.shape))
                
                if test and fold == 0:
                    final_train_tweets = raw_tweets[index_dict['valid'] + index_dict['train']]
                    final_train_labels = labels[index_dict['valid'] + index_dict['train']]
                    final_train_labels_onehot = to_categorical(final_train_labels)
                    final_train_char_vectors = transform_char_seq_with_padding(final_train_tweets, max_len=280, direction='post')
                    print("final_train_char_vectors size {}, final_train_labels size {}".format(final_train_char_vectors.shape, final_train_labels.shape))
                    test_model = model_select(mt, CONV_params, None, None)
                else:
                    # split train test data
                    train_tweets = raw_tweets[index_dict['train']]
                    train_labels = labels[index_dict['train']]
                    train_labels_onehot = to_categorical(train_labels)
                    
                    valid_tweets = raw_tweets[index_dict['valid']]
                    valid_labels = labels[index_dict['valid']]
                    valid_labels_onehot = to_categorical(valid_labels)

                    # vectorize tweets
                    train_char_vectors = transform_char_seq_with_padding(train_tweets, max_len=280, direction='post')
                    valid_char_vectors = transform_char_seq_with_padding(valid_tweets, max_len=280, direction='post')
                    print("train_char_vectors size {}, train_label size {}".format(train_char_vectors.shape, train_labels.shape))
                    print("valid_char_vectors size {}, valid_label size {}".format(valid_char_vectors.shape, valid_labels.shape))

                    base_model = model_select(mt, CONV_params, None, None)
            
            elif mt == 'char_expan':
                clean_tweets, hl_expanded_tweets, labels, hl_term_features, hl_cluster_features = prepare_feature_expan_data_no_matrix(general_params, CONV_params, wordvec_params)
                
                test_tweets = raw_tweets[index_dict['test']]
                test_labels = labels[index_dict['test']]
                test_labels_onehot = to_categorical(test_labels)
                test_term_features = hl_term_features[index_dict['test']]
                test_cluster_features = hl_cluster_features[index_dict['test']]
                test_char_vectors = transform_char_seq_with_padding(test_tweets, max_len=280, direction='post')
                print("test_char_vectors size {}, test_label size {}".format(test_char_vectors.shape, test_labels.shape))

                if test and fold == 0:
                    final_train_tweets = raw_tweets[index_dict['valid'] + index_dict['train']]
                    final_train_labels = labels[index_dict['valid'] + index_dict['train']]
                    final_train_labels_onehot = to_categorical(final_train_labels)
                    final_train_term_features = hl_term_features[index_dict['valid'] + index_dict['train']]
                    final_train_cluster_features = hl_cluster_features[index_dict['valid'] + index_dict['train']]
                    final_train_char_vectors = transform_char_seq_with_padding(final_train_tweets, max_len=280, direction='post')
                    print("final_train_char_vectors size {}, final_train_label size {}".format(final_train_char_vectors.shape, final_train_labels.shape))
                    
                    test_model = model_select(mt, CONV_params, None, None)
                else:
                    train_tweets = raw_tweets[index_dict['train']]
                    train_labels = labels[index_dict['train']]
                    train_labels_onehot = to_categorical(train_labels)
                    train_term_features = hl_term_features[index_dict['train']]
                    train_cluster_features = hl_cluster_features[index_dict['train']]

                    valid_tweets = raw_tweets[index_dict['valid']]
                    valid_labels = labels[index_dict['valid']]
                    valid_labels_onehot = to_categorical(valid_labels)
                    valid_term_features = hl_term_features[index_dict['valid']]
                    valid_cluster_features = hl_cluster_features[index_dict['valid']]

                    # vectorize tweets
                    train_char_vectors = transform_char_seq_with_padding(train_tweets, max_len=280, direction='post')
                    valid_char_vectors = transform_char_seq_with_padding(valid_tweets, max_len=280, direction='post')
                    
                    print("train_char_vectors size {}, train_label size {}".format(train_char_vectors.shape, train_labels.shape))
                    print("valid_char_vectors size {}, valid_label size {}".format(valid_char_vectors.shape, valid_labels.shape))
                    
                    base_model = model_select(mt, CONV_params, None, None)

            if mt == 'word_expan_concat':
                call_back_valid = PredictOnEpoch({'main_input':valid_word_vectors, 
                                                  'term_feature':valid_term_features, 
                                                  'cluster_feature':valid_cluster_features}, valid_labels, 
                                                  general_params['every_n_epoch'], epoch_valid_labels, 
                                                  epoch_valid_preds)
                base_model.fit({'main_input':train_word_vectors, 
                                'term_feature':train_term_features, 
                                'cluster_feature':train_cluster_features},
                                train_labels_onehot, epochs=general_params["epochs"],
                                batch_size=CONV_params["batch_size"], callbacks=[call_back_valid],
                                verbose=CONV_params['verbose'], class_weight=CONV_params['class_weight'])
                if test and fold == 0:
                    call_back_test = PredictOnEpoch({'main_input':test_word_vectors, 
                                                      'term_feature':test_term_features, 
                                                      'cluster_feature':test_cluster_features}, 
                                                      test_labels, general_params['every_n_epoch'], 
                                                      epoch_test_labels, epoch_test_preds)
                    test_model.fit({'main_input':final_train_word_vectors, 
                                    'term_feature':final_train_term_features, 
                                    'cluster_feature':final_train_cluster_features},
                                    final_train_labels_onehot, epochs=general_params["epochs"],
                                    batch_size=CONV_params["batch_size"],callbacks=[call_back_test],
                                    verbose=CONV_params['verbose'], class_weight=CONV_params['class_weight'])
            elif mt == 'char_wide':
                call_back_valid = PredictOnEpoch({'main_input':valid_char_vectors}, 
                                                  valid_labels, general_params['every_n_epoch'], 
                                                  epoch_valid_labels, epoch_valid_preds)
                base_model.fit({'main_input':train_char_vectors}, 
                                train_labels_onehot, epochs=general_params["epochs"],
                                batch_size=CONV_params["batch_size"],callbacks=[call_back_valid],
                                verbose=CONV_params['verbose'], class_weight=CONV_params['class_weight'])
                if test and fold == 0:
                    call_back_test = PredictOnEpoch({'main_input':test_char_vectors}, 
                                                    test_labels, general_params['every_n_epoch'], 
                                                    epoch_test_labels, epoch_test_preds)
                    test_model.fit({'main_input':final_train_char_vectors}, 
                                    final_train_labels_onehot, epochs=general_params["epochs"],
                                    batch_size=CONV_params["batch_size"],callbacks=[call_back_test],
                                    verbose=CONV_params['verbose'], class_weight=CONV_params['class_weight'])
            elif mt == 'char_expan':
                call_back_valid = PredictOnEpoch({'main_input':valid_char_vectors, 
                                                  'term_feature':valid_term_features, 
                                                  'cluster_feature':valid_cluster_features}, valid_labels, 
                                                  general_params['every_n_epoch'], 
                                                  epoch_valid_labels, epoch_valid_preds)
                base_model.fit({'main_input':train_char_vectors, 'term_feature':train_term_features, 
                                'cluster_feature':train_cluster_features},
                                train_labels_onehot, epochs=general_params["epochs"],
                                batch_size=CONV_params["batch_size"], callbacks=[call_back_valid],
                                verbose=CONV_params['verbose'], class_weight=CONV_params['class_weight'])
                if test and fold == 0:
                    call_back_test = PredictOnEpoch({'main_input':test_char_vectors, 
                                                    'term_feature':test_term_features, 
                                                    'cluster_feature':test_cluster_features}, 
                                                    test_labels, general_params['every_n_epoch'], 
                                                    epoch_test_labels, epoch_test_preds)
                    test_model.fit({'main_input':final_train_char_vectors, 
                                    'term_feature':final_train_term_features, 
                                    'cluster_feature':final_train_cluster_features},
                                    final_train_labels_onehot, epochs=general_params["epochs"],
                                    batch_size=CONV_params["batch_size"],callbacks=[call_back_test],
                                    verbose=CONV_params['verbose'], class_weight=CONV_params['class_weight'])
            # base_preds = base_model.predict(test_vectors)
            # base_pred_labels = np.argmax(base_preds, axis=1)
            if test and fold == 0:
                testing_labels.append(epoch_test_labels)
                testing_preds.append(epoch_test_preds)
                del(test_model)
            
            fold_valid_labels[fold].append(epoch_valid_labels)
            fold_valid_preds[fold].append(epoch_valid_preds)
            del(base_model)     
            K.clear_session()

        if test and fold == 0:
            testing_labels = np.asarray(testing_labels)
            testing_preds = np.asarray(testing_preds)
            testing_results = (testing_labels, testing_preds)
            with open(test_result_filename, 'wb') as outf:
                pickle.dump(testing_results, outf)
            
        fold_valid_labels[fold] = np.asarray(fold_valid_labels[fold])
        fold_valid_preds[fold] = np.asarray(fold_valid_preds[fold])
        fold_results = (fold_valid_labels[fold], fold_valid_preds[fold])
        all_fold_results[fold] = fold_results
        with open(epoch_result_filename, 'wb') as outf:
            pickle.dump(fold_results, outf)
        fold += 1


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-i", "--index", dest="index_number",
                    help="index number of the model and related data", required=True)
    parser.add_argument("-t", "--test", dest="test",
                    help="whether report test results or valid results", required=True)
    parser.add_argument("-pos", "--positive", dest="pos",
                    help="positive label ratio", required=True)
    parser.add_argument("-neg", "--negative", dest="neg",
                    help="negative label ratio", required=True)
    args = parser.parse_args()

    index_number = args.index_number
    pos = float(args.pos)
    neg = float(args.neg)
    test = bool(int(args.test))

    model_types = ['char_expan', 'char_expan', 'char_expan', 'char_wide', 'char_wide', 'char_wide', 'word_expan_concat', 'word_expan_concat', 'word_expan_concat']
    model_type_index = {'char_expan':[0,1,2], 'char_wide':[3,4,5], 'word_expan_concat':[6,7,8]}
    # model_types = ['char_expan', 'char_expan', 'char_wide', 'char_wide', 'word_expan_concat', 'word_expan_concat']
    # model_type_index = {'char_expan':[0,1], 'char_wide':[2,3], 'word_expan_concat':[4,5]}

    # got config ans set class weight
    general_params = get_general_params()
    general_params['label_ratio'] = [neg, pos]
    if general_params['label_ratio'] == [0.5,0.5]:
        cw = {0:1.0, 1:1.0}
    elif general_params['label_ratio'] == [0.6,0.4]:
        cw = {0:0.8, 1:1.2}
    elif general_params['label_ratio'] == [0.7,0.3]:
        cw = {0:0.6, 1:1.4}
    elif general_params['label_ratio'] == [0.8,0.2]:
        cw = {0:0.4, 1:1.6}
    elif general_params['label_ratio'] == [0.9,0.1]:
        cw = {0:0.3, 1:2.7}

    print('train model')
    paralle_ensemble(general_params, cw, index_number, model_types, test)


