import numpy as np
import os, pickle
from argparse import ArgumentParser

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

from training_params import get_general_params
from training_utils import *

# process testing results
# ensemble pred_labels or preds
# return measures of each epoch
def process_test_results(labels, general_params, results_save_path, kfold_index_lists, model_types, label):
    folds = range(len(kfold_index_lists))
    # read testing data
    epoch_log_filename = results_save_path + "models_all_test_results.pik"
    print('reading file {}'.format(epoch_log_filename))
    with open(epoch_log_filename, 'rb') as inf:
        all_fold_results = pickle.load(inf)
    # ensemble preds
    fold = 0
    gt_labels = labels[kfold_index_lists[fold]['test']]
    # shape [models, epochs, n_samples]
    # shape [models, epochs, n_samples, 2]
    fold_test_labels, fold_test_preds = all_fold_results
    if not label:
        # shape [epochs, n_samples, 2]
        fold_test_ensemble_preds = np.mean(fold_test_preds, axis=0)
        # shape [epochs, n_samples]
        fold_test_ensemble_labels = np.argmax(fold_test_ensemble_preds, axis=2)
    else:
        thres = len(model_types) / 2
        # shape [epochs, n_samples]
        fold_test_ensemble_labels = np.where(np.sum(fold_test_labels, axis=0) > thres, 1, 0)
    # compute measures
    measures = []
    for epoch in range(fold_test_ensemble_preds.shape[0]):
        acc = accuracy_score(gt_labels, fold_test_ensemble_labels[epoch])
        pos_prec = precision_score(gt_labels, fold_test_ensemble_labels[epoch], pos_label=1)
        neg_prec = precision_score(gt_labels, fold_test_ensemble_labels[epoch], pos_label=0)
        pos_recall = recall_score(gt_labels, fold_test_ensemble_labels[epoch], pos_label=1)
        neg_recall = recall_score(gt_labels, fold_test_ensemble_labels[epoch], pos_label=0)
        pos_f1 = f1_score(gt_labels, fold_test_ensemble_labels[epoch], pos_label=1)
        neg_f1 = f1_score(gt_labels, fold_test_ensemble_labels[epoch], pos_label=0)
        measures.append((acc, pos_prec, neg_prec, pos_recall, neg_recall, pos_f1, neg_f1))
    
    # shape [epochs, 7]
    measures = np.asarray(measures)

    return measures
    
# process validation results
# ensemble pred_labels or preds
# return measures of each epoch
def process_valid_results(labels, general_params, results_save_path, kfold_index_lists, model_types, label):
    folds = range(len(kfold_index_lists))
    # read results
    epoch_log_filename = results_save_path + "models_all_valid_results.pik"
    print('reading file {}'.format(epoch_log_filename))
    with open(epoch_log_filename, 'rb') as inf:
        all_fold_results = pickle.load(inf)
    # process results
    all_folds_measures = []
    for fold in folds:
        gt_labels = labels[kfold_index_lists[fold]['valid']]
        # shape [models, epochs, n_samples]
        # shape [models, epochs, n_samples, 2]
        fold_valid_labels, fold_valid_preds = all_fold_results[fold]
        if not label:
            # shape [epochs, n_samples, 2]
            fold_valid_ensemble_preds = np.mean(fold_valid_preds, axis=0)
            # shape [epochs, n_samples]
            fold_valid_ensemble_labels = np.argmax(fold_valid_ensemble_preds, axis=2)
        else:
            thres = len(model_types) / 2
            # shape [epochs, n_samples]
            fold_valid_ensemble_labels = np.where(np.sum(fold_valid_labels, axis=0) > thres, 1, 0)
        # compute measures
        measures = []
        for epoch in range(fold_valid_ensemble_labels.shape[0]):
            acc = accuracy_score(gt_labels, fold_valid_ensemble_labels[epoch])
            pos_prec = precision_score(gt_labels, fold_valid_ensemble_labels[epoch], pos_label=1)
            neg_prec = precision_score(gt_labels, fold_valid_ensemble_labels[epoch], pos_label=0)
            pos_recall = recall_score(gt_labels, fold_valid_ensemble_labels[epoch], pos_label=1)
            neg_recall = recall_score(gt_labels, fold_valid_ensemble_labels[epoch], pos_label=0)
            pos_f1 = f1_score(gt_labels, fold_valid_ensemble_labels[epoch], pos_label=1)
            neg_f1 = f1_score(gt_labels, fold_valid_ensemble_labels[epoch], pos_label=0)
            measures.append((acc, pos_prec, neg_prec, pos_recall, neg_recall, pos_f1, neg_f1))
        all_folds_measures.append(measures)

    # average across folds
    # shape [folds, epochs, 7]
    all_folds_measures = np.asarray(all_folds_measures)
    # shape [epochs, 7]
    average_measures = np.mean(all_folds_measures, axis=0)

    return average_measures


def read_fold_results(general_params, model_types, index_number, model_type_index, test, use_label):
    # read data
    raw_tweets, clean_tweets, labels = read_labeled_data(general_params, shuffle=False)

    # read index
    kfold_index_lists = get_RS_index(raw_tweets, labels, general_params, index_number, force_new=False)

    results_save_path = general_params['results_home'] + 'ensemble_ml_{}_p_{}_n_{}_index_{}/'.format(model_types,
        general_params['label_ratio'][1], general_params['label_ratio'][0], index_number)

    if test:
        process_test_results(labels, general_params, results_save_path, kfold_index_lists, model_types, use_label)
    else:
        process_valid_results(labels, general_params, results_save_path, kfold_index_lists, model_types, use_label)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-i", "--index", dest="index_number",
                    help="index number of the model and related data", required=True)
    parser.add_argument("-t", "--test", dest="test",
                    help="whether report test results or valid results", required=True)
    parser.add_argument("-l", "--use_label", dest="use_label",
                    help="ensemble with labels or preds", required=True)
    parser.add_argument("-pos", "--positive", dest="pos",
                    help="positive label ratio", required=True)
    parser.add_argument("-neg", "--negative", dest="neg",
                    help="negative label ratio", required=True)
    args = parser.parse_args()

    index_number = args.index_number
    use_label = bool(int(args.use_label))
    test = bool(int(args.test))
    pos = float(args.pos)
    neg = float(args.neg)

    model_types = ['svm', 'svm', 'svm', 'rf', 'rf', 'rf', 'nb', 'nb','nb']
    model_type_index = {'svm':[0,1,2], 'rf':[3,4,5], 'nb':[6,7,8]}

    general_params = get_general_params(debug)
    general_params['label_ratio'] = [neg, pos]
    
    print("reading results")
    read_fold_results(general_params, model_types, index_number, model_type_index, test, use_label)