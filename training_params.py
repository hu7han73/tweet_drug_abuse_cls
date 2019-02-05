# def get_SVM_params():
#     params = {
#         'prob':True,
#         'class_weight':'balanced', # balanced or None
#         'C':{0.1:0.5, 0.2:0.5, 0.3:0.5, 0.4:1.0, 0.5:1.0},
#         'gamma':{0.1:0.01, 0.2:0.05, 0.3:0.05, 0.4:0.05, 0.5:0.1}, 
#         'kernel':'rbf', 
#     }
#     return params

def get_SVM_params():
    params = {
        'prob':True,
        'class_weight':'balanced', # balanced or None
        'C':{0.1:1.0, 0.2:1.0, 0.3:1.0, 0.4:1.0, 0.5:2.0},
        'gamma':{0.1:0.05, 0.2:0.05, 0.3:0.05, 0.4:0.05, 0.5:0.05}, 
        'kernel':'rbf', 
    }
    return params

def get_RF_params():
    params = {
        'n_estimators':500,
        'criterion':'gini',
        'max_depth':20,
        'min_samples_split':2,
        'min_samples_leaf':1,
        'min_weight_fraction_leaf':0.0, 
        'max_features':0.4, 
        'max_leaf_nodes':50,
        'min_impurity_decrease':0.0, 
        'bootstrap':True, 
        'n_jobs':8, 
        'warm_start':False, 
        'class_weight':'balanced'
    }
    return params

def get_NB_params():
    params = {
        'alpha':1.0,
        'fit_prior':True,
        'gaussian':False
    }
    return params
# old CNN params
def get_CONV_params():
    params = {
        'batch_size':64, 
        'verbose':0, 
        "conv_epochs":100, 
        'sequence_length':20, 
        'dropout_prob':[0.2,0.5], 
        "num_filters":20,
        "hidden_dim":500, 
        'n_gram_list':[2,3,4],
        'use_keras_embedding':False,
    }
    return params

#============= new CNN params ============
def get_CONV_deep_params():
    params = {
        # model params
        'sequence_length':30,
        'use_keras_embedding':False,
        'conv_layers':[(128, 3, 2),(128, 3, 2)], # num_filters, kernal_size(n_gram), pool_size
        'cnn_dropout':0.1,
        'fc_layers':[(1024, 0.5),(1024, 0.5)],
        # training params
        'batch_size':64, 
        'verbose':2, 
        "conv_epochs":50, 
    }
    return params

def get_CONV_wide_params():
    params = {
        # model params
        'sequence_length':30,
        'use_keras_embedding':False,
        'n_gram_list':[2,3,4,5],
        'num_filters':128,
        'cnn_dropout':0.1,
        'fc_layers':[(1024, 0.5),(1024, 0.5)],
        # training params
        'batch_size':64, 
        'verbose':2,  
        "conv_epochs":50, 
    }
    return params

def get_CONV_wide_deep_params():
    params = {
        # model params
        'sequence_length':40,
        'use_keras_embedding':False,
        'cnn_layers':[{'ngram':2, 'num_filters':(128, 128), 'pool_size':(2,2)},
            {'ngram':3, 'num_filters':(128, 128), 'pool_size':(2,2)},
            {'ngram':4, 'num_filters':(128, 128), 'pool_size':(2,2)},
            {'ngram':5, 'num_filters':(128, 128), 'pool_size':(2,2)}],
        #'conv_layers':[(128, 3, 2),(128, 3, 2)], # num_filters, kernal_size(n_gram), pool_size
        'term_feature_len':4,
        'cluster_feature_len':150,
        'cnn_dropout':0.0,
        'fc_layers':[(1024, 0.5),(1024, 0.5)],
        # training params
        'batch_size':64, 
        'verbose':2, 
        "conv_epochs":20, 
    }
    return params

def get_CONV_wide_deep_syn_expan_params():
    params = {
        # model params
        'sequence_length':30,
        'syn_expan_length':20,
        'use_keras_embedding':False,
        'cnn_layers':[{'ngram':2, 'num_filters':(128, 128), 'pool_size':(2,2)},
            {'ngram':3, 'num_filters':(128, 128), 'pool_size':(2,2)},
            {'ngram':4, 'num_filters':(128, 128), 'pool_size':(2,2)},
            {'ngram':5, 'num_filters':(128, 128), 'pool_size':(2,2)}],
        'syn_cnn_layers':[{'ngram':2, 'num_filters':(128, 128), 'pool_size':(2,2)},
            {'ngram':3, 'num_filters':(128, 128), 'pool_size':(2,2)},
            {'ngram':4, 'num_filters':(128, 128), 'pool_size':(2,2)}],
        #'conv_layers':[(128, 3, 2),(128, 3, 2)], # num_filters, kernal_size(n_gram), pool_size
        'term_feature_len':4,
        'cluster_feature_len':150,
        'cnn_dropout':0.1,
        'fc_layers':[(1024, 0.5),(1024, 0.5)],
        # training params
        'batch_size':64, 
        'verbose':2,  
        "conv_epochs":50, 
    }
    return params
#===========================================

#============= char CNN params ============
def get_CONV_char_deep_params():
    params = {
        # model params
        'sequence_length':280,
        'vocab_size':70,
        'embedding_size':128,
        'conv_layers':[(256, 7, 3),(256, 7, 3),(256, 3, -1),(256, 3, -1),(256, 3, -1),(256, 3, 3)], # num_filters, kernal_size(n_gram), pool_size
        'fc_layers':[(1024, 0.5),(1024, 0.5)],
        'relu_threshold':1e-6,
        # training params
        'batch_size':64, 
        'verbose':2, 
        "conv_epochs":25,
        'cnn_dropout':0.1,
        'term_feature_len':4,
        'cluster_feature_len':150,
    }
    return params

def get_CONV_char_wide_params():
    params = {
        # model params
        'sequence_length':280,
        'vocab_size':70,
        'embedding_size':128,
        'conv_layers':[(256, 10),(256, 7),(256, 5),(256, 3)], # num_filters, kernal_size(n_gram), pool_size
        'fc_layers':[(1024, 0.1),(1024, 0.1)],
        'relu_threshold':1e-6,
        # training params
        'batch_size':64, 
        'verbose':2, 
        "conv_epochs":25,
        'term_feature_len':4,
        'cluster_feature_len':150,
    }
    return params
#==================================

def get_LSTM_params():
    params = {
        'batch_size':64, 
        'verbose':0, 
        'lstm_epochs':100, 
        'sequence_length':20,
        'dropout_prob':[0.2,0.5], 
        "hidden_dim":500, 
        'sequence':False,
        'use_keras_embedding':False
    }
    return params

def get_general_params():
    path_params = {
        'results_home':'./results/',
        'data_home':'./data/',
        'temp_home':'./temp/',
        'wordvec_home':'./word_vectors/'}

    params = {
        'sample_balance': True,
        'recursive_size': 2000,
        'recursive_sample_size':200,
        'recursive_times':4,
        'confidence_threshold':0.7,
        'kfolds':6,
        'label_ratio':[0.5,0.5],
        'every_n_epoch':1,
        'epochs':50,
        'num_replace':5,
        'word_swap_chance':0.8,

        'hl_weight_rate':1.2,
        'hl_weight_min':0.99,
        'hl_weight_max':1.5,
        'ml_weight_ratio':0.8,
        
        'kfold_filename':'kfold_index_lists.pik',
        
        'hl_syn_expan_dict_filename':'hl_syn_expan_dict_cnn.pik',
        'hl_syn_expan_lists_filename':'hl_syn_expan_lists_cnn.pik',
        'hl_term_feature_filename':'hl_term_feature_vector_cnn.pik',
        'hl_cluster_feature_filename':'hl_cluster_feature_vector_cnn.pik',

        '5k_hl_data_filename':'MT_5k_labels_raw_tweets_clean_tweets_unique.csv',
        '5k_hl_token_filename':'MT_5k_token_lists.pik',
        '5k_hl_tagging_filename':'MT_5k_tagging_results.pik',
        'tweets_countvectorizer_filename':'tweets_countvectorizer.pik',
        'syn_expan_countvectorizer_filename':'syn_expan_countvectorizer.pik',
        'final_labeled_vectors_filename':'final_MT_5k_tweets_vectors.pik',
        'kfold_seed':1024
    }
    params.update(path_params)
    return params

def get_Googlenews_wordvec_params():
    params = {
        'wordmatrix_file':'Googlenews_wordmatrix.pik',
        'word2vec_file':'GoogleNews-vectors-negative300.bin',
        'embedding':'google',
        'embedding_dim':300,
    }
    return params

def get_Glove_wordvec_params():
    params = {
        'wordmatrix_file':'Glove_wordmatrix.pik',
        'word2vec_file':'glove.42B.300d.txt',
        'embedding':'glove',
        'embedding_dim':300,
    }
    return params

def get_Godin_wordvec_params():
    params = {
        'wordmatrix_file':'Godin_wordmatrix.pik',
        'word2vec_file':'Godin_word2vec_twitter_model.bin',
        'embedding':'godin',
        'embedding_dim':400,
    }
    return params

def get_DSM_wordvec_params():
    params = { # word2vec binary with utf8 codec and error=ignore
        'wordmatrix_file':'DSM_wordmatrix.pik',
        'wordmatrix_file_syn_expan':'DSM_wordmatrix_syn_expan.pik',
        'wordmatrix_file_syn_expan2':'DSM_wordmatrix_syn_expan2.pik',
        'word2vec_file':'DSM_word2vec_model.bin',
        'embedding':'dsm',
        'embedding_dim':400,
    }
    return params

def get_DSM_wordvec_syn_expan_params():
    params = { # word2vec binary with utf8 codec and error=ignore
        'wordmatrix_file':'DSM_wordmatrix.pik',
        'wordmatrix_file_syn_expan':'DSM_wordmatrix_syn_expan.pik',
        'wordmatrix_file_syn_expan2':'DSM_wordmatrix_syn_expan2.pik',
        'word2vec_file':'DSM_word2vec_model.bin',
        'embedding':'dsm',
        'embedding_dim':400,
    }
    return params

def get_new_wordvec_params():
    params = { # word2vec binary with utf8 codec and error=ignore
        'wordmatrix_file':'new_wordmatrix.pik',
        'word2vec_file':'word2vec_skip_17_70_3m.pik',
        'embedding':'custom',
        'embedding_dim':300,
    }
    return params

def get_custom_wordvec_params():
    params = { # word2vec binary with utf8 codec and error=ignore
        'wordmatrix_file':'custom_wordmatrix.pik',
        'word2vec_file':'word2vec_all_tweets_skip_nostem.w2v',
        'embedding':'gensim',
        'embedding_dim':300,
    }
    return params

