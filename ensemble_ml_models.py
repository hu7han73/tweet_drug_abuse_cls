import numpy as np
import os, pickle
from argparse import ArgumentParser

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

from training_params import get_SVM_params, get_RF_params, get_NB_params, get_general_params
from training_utils import read_labeled_data, read_hl_vectors, get_index
from read_ml_results import read_fold_results


# select model given string name
def model_select(model_type, model_params, general_params):
    if model_type == 'svm':
        base_model = svm.SVC(C=model_params['C'][general_params['label_ratio'][1]], kernel=model_params['kernel'], 
            gamma=model_params['gamma'][general_params['label_ratio'][1]], probability=True, class_weight=model_params['class_weight'])
    elif model_type == 'rf':
        base_model = RandomForestClassifier(n_estimators=model_params['n_estimators'], 
            criterion=model_params['criterion'], max_depth=model_params['max_depth'], 
            min_samples_split=model_params['min_samples_split'], min_samples_leaf=model_params['min_samples_leaf'], 
            min_weight_fraction_leaf=model_params['min_weight_fraction_leaf'], max_features=model_params['max_features'], 
            max_leaf_nodes=model_params['max_leaf_nodes'], min_impurity_decrease=model_params['min_impurity_decrease'], 
            bootstrap=model_params['bootstrap'], n_jobs=model_params['n_jobs'], warm_start=model_params['warm_start'], 
            class_weight=model_params['class_weight'])
    elif model_type == 'nb':
        base_model = MultinomialNB(alpha=model_params['alpha'], fit_prior=model_params['fit_prior'], 
            class_prior=None)
    else:
        print('model type invalid')
        exit()
    return base_model

def model_param_select(model_type):
    if model_type == 'svm':
        params = get_SVM_params()
    elif model_type == 'rf':
        params = get_RF_params()
    elif model_type == 'nb':
        params = get_NB_params()
    else:
        print('model type invalid')
        exit()
    return params

def run_model(train_vectors, train_labels, test_vectors, test_labels, model_type, model_params, general_params):
    # train and predict
    model = model_select(model_type, model_params, general_params)
    model.fit(train_vectors, train_labels)
    test_preds = model.predict_proba(test_vectors)
    test_pred_labels = model.predict(test_vectors)

    return test_pred_labels, test_preds

# ensemble same types of models
def paralle_ensemble(general_params, index_number, model_types, test):  
    # read data
    raw_tweets, clean_tweets, labels = read_labeled_data(general_params, shuffle=False)
    labeled_vectors = read_hl_vectors(general_params).tocsr()

    # read index
    kfold_index_lists = get_index(raw_tweets, labels, general_params, index_number, force_new=False)

    # if doing training for validation, do k-fold and save all preds for further ensenmble
    if not test:
        # prepare dict of results for ensemble
        fold_test_labels = {}
        fold_valid_labels = {}
        fold_test_preds = {}
        fold_valid_preds = {}
        all_fold_results = {}

        # prepare folder to save results
        results_save_path = general_params['results_home'] + 'ensemble_ml_{}_p_{}_n_{}_index_{}/'.format(model_types,
                    general_params['label_ratio'][1], general_params['label_ratio'][0], index_number)
        if not os.path.exists(results_save_path):
            os.mkdir(results_save_path)

        fold_log_filename = results_save_path + "models_all_valid_results.pik"
        # train model one by one
        fold = 0
        for index_dict in kfold_index_lists:
            # prepare filename for each fold
            fold_test_labels[fold] = []
            fold_valid_labels[fold] = []
            fold_test_preds[fold] = []
            fold_valid_preds[fold] = []

            # train each model       
            model_count = 0
            for mt in model_types:
                model_count += 1
                print('fold {}, model {}'.format(fold, model_count))
                model_params = model_param_select(mt)
                # split training and testing data
                # if doing validation (parameter tuning), then use training dataset of this fold
                # if doing testing, then use training and validation
                train_vectors = labeled_vectors[index_dict['train']]
                test_vectors = labeled_vectors[index_dict['valid']]
                train_labels = labels[index_dict['train']]
                test_labels = labels[index_dict['valid']]

                valid_pred_labels, valid_preds = run_model(train_vectors, train_labels, test_vectors, test_labels, mt, model_params, general_params)

                fold_test_labels[fold].append(valid_pred_labels)
                fold_test_preds[fold].append(valid_preds)
            
            # fold_test_labels[fold] = np.asarray(fold_test_labels[fold])
            # fold_test_preds[fold] = np.asarray(fold_test_preds[fold])
            all_fold_results[fold] = (fold_test_labels[fold], fold_test_preds[fold])
            fold += 1
        
        with open(fold_log_filename, 'wb') as outf:
            pickle.dump(all_fold_results, outf)

    # if doing testing, then train with everything and do test, save only test results
    else:
        # prepare folder to save results
        results_save_path = general_params['results_home'] + 'ensemble_ml_{}_p_{}_n_{}_index_{}/'.format(model_types,
                    general_params['label_ratio'][1], general_params['label_ratio'][0], index_number)
        if not os.path.exists(results_save_path):
            os.mkdir(results_save_path)

        fold_log_filename = results_save_path + "models_all_test_results.pik"
        # train each model       
        model_count = 0
        index_dict = kfold_index_lists[0]
        all_test_results = []
        for mt in model_types:
            model_count += 1
            print("model {}".format(model_count))
            model_params = model_param_select(mt)

            # split training and testing data
            # if doing validation (parameter tuning), then use training dataset of this fold
            # if doing testing, then use training and validation
            train_vectors = labeled_vectors[index_dict['train'] + index_dict['valid']]
            test_vectors = labeled_vectors[index_dict['test']]
            train_labels = labels[index_dict['train'] + index_dict['valid']]
            test_labels = labels[index_dict['test']]

            # train and predict
            model = model_select(mt, model_params, general_params)
            model.fit(train_vectors, train_labels)
            test_preds = model.predict_proba(test_vectors)
            test_pred_labels = model.predict(test_vectors)

            test_pred_labels, test_preds = run_model(train_vectors, train_labels, test_vectors, test_labels, mt, model_params, general_params)

            all_test_results.append((test_pred_labels, test_preds))

        with open(fold_log_filename, 'wb') as outf:
            pickle.dump(all_test_results, outf)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-i", "--index", dest="index_number",
                    help="index number of the model and related data", required=True)
    parser.add_argument("-d", "--debug", dest="debug",
                    help="whether debug (run on personal pc)", required=True)
    parser.add_argument("-t", "--test", dest="test",
                    help="whether report test results or valid results", required=True)
    parser.add_argument("-pos", "--positive", dest="pos",
                    help="positive label ratio", required=True)
    parser.add_argument("-neg", "--negative", dest="neg",
                    help="negative label ratio", required=True)
    args = parser.parse_args()

    index_number = args.index_number
    debug = int(args.debug)
    test = bool(int(args.test))
    pos = float(args.pos)
    neg = float(args.neg)

    model_types = ['svm', 'svm', 'svm', 'rf', 'rf', 'rf', 'nb', 'nb','nb']
    model_type_index = {'svm':[0,1,2], 'rf':[3,4,5], 'nb':[6,7,8]}

    # model_types = ['svm', 'svm', 'svm']
    # model_type_index = {'svm':[0,1,2]}

    general_params = get_general_params()
    general_params['label_ratio'] = [neg, pos]
    #general_params['label_ratio'] = [0.5,0.5]

    print('training')
    paralle_ensemble(general_params, index_number, model_types, test)
    # read_fold_results(general_params, model_types, debug, index_number, model_type_index, test=test, ensemble=False)
    # read_fold_results(general_params, model_types, debug, index_number, model_type_index, test=test, ensemble=True)