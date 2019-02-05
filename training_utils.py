import os, pickle, re
import numpy as np
import pandas as pd
from collections import Counter
import itertools
import preprocessor as p

def read_pickle(filename):
    with open(filename, 'rb') as inf:
        obj = pickle.load(inf)
    return obj

def dump_pickle(obj, filename):
    with open(filename, 'wb') as outf:
          pickle.dump(obj, outf)

# read the csv file that contains raw and clean tweets
# although clean tweets are not used here
def read_labeled_data(general_params, shuffle=False):
    print('read labeled 5k dataset')
    data = pd.read_csv(general_params['data_home'] + general_params['5k_hl_data_filename'], na_filter=False, encoding='utf-8', engine='python')

    raw_tweet = np.asarray([t for t in data.raw_tweet])
    clean_tweet = np.asarray([t for t in data.clean_tweet])
    labels = np.asarray([t for t in data.final_label])

    idx = list(range(raw_tweet.shape[0]))
    if shuffle:
        np.random.shuffle(idx)
        raw_tweet = raw_tweet[idx]
        clean_tweet = clean_tweet[idx]
        labels = labels[idx]
    print(raw_tweet.shape)
    return raw_tweet, clean_tweet, labels

# build vocabulary base on a set of tweets
def build_vocab(token_lists):
    print('building vocab')
    # A better looking solution with Counter and itertools.chain
    word_counts = Counter(itertools.chain(*token_lists))
    word_frequency_list = word_counts.most_common()
    print(word_frequency_list[0:5])
    # inv_vocab is now vocab_size+1, no need +1 later
    inv_vocab = ['__VOCABPLACEHOLDER__'] + [x[0] for x in word_frequency_list] # [0] position is a placeholder
    vocab = {x : i+1 for i,x in enumerate(inv_vocab[1:])} # reserve 0 for padding
    vocab['__VOCABPLACEHOLDER__'] = 0
    return vocab, inv_vocab, word_frequency_list


def read_hl_vectors(general_params):
    final_hl_vectors_filename = general_params['temp_home'] + general_params['final_labeled_vectors_filename']
    with open(final_hl_vectors_filename, 'rb') as outf:
        final_labeled_vectors = pickle.load(outf)
    return final_labeled_vectors

def read_nl_vectors(general_params):
    final_nl_vectors_filename = general_params['temp_home'] + general_params['final_unlabeled_vectors_filename']
    with open(final_nl_vectors_filename, 'rb') as outf:
        final_unlabeled_vectors = pickle.load(outf)
    return final_unlabeled_vectors
    

def read_clean_data(general_params, shuffle=False):
    data = pd.read_csv(general_params['data_home'] + general_params['5k_hl_data_filename'], na_filter=False, encoding='utf-8', engine='python')

    clean_tweets = np.asarray([t.split() for t in data.clean_tweet])
    labels = np.asarray([t for t in data.final_label])

    idx = list(range(clean_tweets.shape[0]))
    if shuffle:
        np.random.shuffle(idx)
        clean_tweets = clean_tweets[idx]
        labels = labels[idx]
    print(clean_tweets.shape)
    return clean_tweets, labels

def get_index(tweets, labels, general_params, index_number, force_new=False):
    file_path = '{}kfold_{}_ratio_p{}-n{}_{}_{}'.format(general_params['temp_home'], general_params['kfolds'], general_params['label_ratio'][1], general_params['label_ratio'][0], index_number, general_params['kfold_filename'])
    if force_new or not os.path.isfile(file_path):
        print('Creating new index profile, ratio: {}-{}, number: {}'.format(general_params['label_ratio'][1], general_params['label_ratio'][0], index_number))
        # create new index profile
        total_folds = general_params['kfolds'] + 1
        unique_labels = [0, 1]
        index = list(range(tweets.shape[0]))
        label_index = np.asarray([[j for j in index if labels[j] == 0], [j for j in index if labels[j] == 1]])
        # max size that a test dataset can take (round to 10)
        max_test_size = int(int(min([int(len(label_index[0]) / (total_folds * general_params['label_ratio'][0])) - 1, int(len(label_index[1]) / (total_folds * general_params['label_ratio'][1])) - 1]) / 10) * 10)
        print('max test (fold) size: {}'.format(max_test_size))
        total_size = max_test_size * total_folds
        print('total size: {}'.format(total_size))
        class_sizes = [int(total_size * general_params['label_ratio'][0]), int(total_size * general_params['label_ratio'][1])]
        print("positive size: {}, negative size: {}".format(class_sizes[1], class_sizes[0]))
        # sample all tweets that is going into a batch
        batch_index = {}
        for label in unique_labels:
            batch_index[label] = np.random.choice(label_index[label], class_sizes[label], replace=False)
            np.random.shuffle(batch_index[label])
        # sample folds
        kfold_raw_index_list = []
        for i in range(total_folds):
            fold_index = {}
            for label in unique_labels:
                if batch_index[label].shape[0] >= int(max_test_size * general_params['label_ratio'][label]):
                    fold_index[label] = np.random.choice(batch_index[label], int(max_test_size * general_params['label_ratio'][label]), replace=False)
                else:
                    fold_index[label] = batch_index[label]
                batch_index[label] = np.setdiff1d(batch_index[label], fold_index[label])
            combined_fold_index = np.concatenate([fold_index[l] for l in unique_labels])
            kfold_raw_index_list.append(combined_fold_index)        
        # arrange each fold's training and testing data
        kfold_train_test_index_list = []
        for i in range(1, total_folds):
            fold_train_test_index = {}
            fold_train_test_index['valid'] = kfold_raw_index_list[i].tolist()
            fold_train_test_index['test'] = kfold_raw_index_list[0].tolist()
            fold_train_test_index['train'] = (np.concatenate([kfold_raw_index_list[x] for x in range(1, total_folds) if x != i]).tolist())
            print(fold_train_test_index)
            kfold_train_test_index_list.append(fold_train_test_index)
        # write index to file
        with open(file_path, 'wb') as outf:
            pickle.dump(kfold_train_test_index_list, outf)
            print('kfold file saved')
    else:
        print('Reading new index profile, kfolds {}, ratio: p{}-n{}, number: {}'.format(general_params['kfolds'], general_params['label_ratio'][1], general_params['label_ratio'][0], index_number))
        # read existing profile
        with open(file_path,'rb') as inf:
            kfold_train_test_index_list = pickle.load(inf)
            print('kfold file loaded')
            test_size = len(kfold_train_test_index_list[0]['test'])
            print('test_size: {}'.format(test_size))
            total_size = len(kfold_train_test_index_list[0]['test']) + len(kfold_train_test_index_list[0]['valid']) + len(kfold_train_test_index_list[0]['train'])
            print('total size: {}'.format(total_size))
    return kfold_train_test_index_list

# pre-process (without stemming) a raw tweet and return a list of words
def raw_tweet_prep_test(raw_tweet, stopwords, html_re, space_replace_re, repeating_re, single_char_re):
    tweet_tokenized = html_re.sub(' ', raw_tweet)
    tweet_tokenized = p.tokenize(tweet_tokenized.lower().replace('\n',' '))
    tweet_tokenized = space_replace_re.sub(' ', tweet_tokenized)
    tweet_tokenized = repeating_re.sub(r"\1", tweet_tokenized)
    #raw_tweet = ' '.join(raw_tweet)
    
    #tweet_tokenized = single_char_re.sub(' ', tweet_tokenized)
    tweet_tokenized = tweet_tokenized.strip().split()
    words = [w for w in tweet_tokenized if w not in stopwords]
    if len(words) > 1:
        return words
    else:
        raise Exception("Input tweet too short")

# pre-process (with stemming) a raw tweet and return a list of words
def raw_tweet_prep_stem_test(raw_tweet, stopwords, stemmer, html_re, space_replace_re, repeating_re, single_char_re):
    # remove some more things ('s, 'm, 't, html symbol, other non english char, and repeating expression)
    tweet_tokenized = html_re.sub(' ', raw_tweet)
    print(tweet_tokenized)
    tweet_tokenized = p.tokenize(tweet_tokenized.lower().replace('\n',' '))
    print(tweet_tokenized)
    tweet_tokenized = space_replace_re.sub(' ', tweet_tokenized)
    print(tweet_tokenized)
    tweet_tokenized = repeating_re.sub(r"\1", tweet_tokenized)
    print(tweet_tokenized)
    # tokenize and replace url with 'URL', numbers with 'NUMBER' and 'EMOJi'
    #tweet_tokenized = single_char_re.sub(' ', tweet_tokenized)
    tweet_tokenized = tweet_tokenized.strip().split()
    print(tweet_tokenized)
    words = [stemmer.stem(w) for w in tweet_tokenized if w not in stopwords]
    if len(words) > 1:
        return words
    else:
        raise Exception("Input tweet too short")

# prettify a raw tweet
def prettify_raw_tweet(raw_tweet):
    html_re = re.compile(r"&#?\w+;")
    new_tweet = html_re.sub(' ', raw_tweet)
    new_tweet = ' '.join(new_tweet.split())
    return new_tweet

# read a list of stop words
def read_stopwords(filename):
    stopwords = []
    with open(filename, 'r', encoding='utf-8') as inf:
        for line in inf:
            stopwords.append(line.strip())
    stopwords = set(stopwords)
    return stopwords