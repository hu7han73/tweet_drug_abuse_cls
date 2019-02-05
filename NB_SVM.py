from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm

from keras.preprocessing.sequence import pad_sequences
from collections import Counter
import itertools
import gensim
from preTrainedReader import PreTrainedVectorReader as ptvr
from scipy.sparse import csr_matrix, lil_matrix, save_npz, load_npz

import numpy as np
from numpy import linalg as la
import os,sys,pickle

def if_train_load_tfidf(all_tweets, vect_file):
    if os.path.isfile(vect_file):
        print("tfidf_vect found, load it")
        with open(vect_file, 'rb') as inf:
            vect = pickle.load(inf)
        return vect
    else:
        print("No tfidf_vect found, train a new one and save")
        vect = TfidfVectorizer()
        vect.fit(all_tweets)
        with open(vect_file, 'wb') as outf:
            pickle.dump(vect, outf)
        return vect

def if_train_load_NB(train_vectors, train_labels, NB_model_file, NB_params):
    if os.path.isfile(NB_model_file):
        print("NB model found, load it")
        with open(NB_model_file, 'rb') as inf:
            model = pickle.load(inf)
        return model
    else:
        print("No NB model found, train a new one and save")
        model = MultinomialNB(alpha=NB_params['alpha'])
        model.fit(train_vectors, train_labels)
        with open(NB_model_file, 'wb') as outf:
            pickle.dump(model, outf)
        return model

def if_train_load_SVM(train_vectors, train_labels, SVM_model_file, SVM_params):
    if os.path.isfile(SVM_model_file):
        print("SVM model found, load it")
        with open(SVM_model_file, 'rb') as inf:
            model = pickle.load(inf)
        return model
    else:
        print("No SVM model found, train a new one and save")
        model = svm.SVC(C=SVM_params['C'], kernel=SVM_params['kernel'],
                gamma=SVM_params['gamma'], probability=SVM_params['prob'],
                class_weight=SVM_params['class_weight'])
        model.fit(train_vectors, train_labels)
        with open(SVM_model_file, 'wb') as outf:
            pickle.dump(model, outf)
        return model

def train_SVM(train_vectors, train_labels, SVM_params):
    print("Training new SVM without saving")
    model = svm.SVC(C=SVM_params['C'], kernel=SVM_params['kernel'],
            gamma=SVM_params['gamma'], probability=SVM_params['prob'],
            class_weight=SVM_params['class_weight'])
    model.fit(train_vectors, train_labels)
    return model


#================== Utils ==============

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
    
    # determine infrequent word vec
    infreq_word_size = round(vocab_size / 100.0 * 1.0)
    infreq_word_freq = word_frequency_list[-infreq_word_size][1]
    print(infreq_word_freq)
    infreq_word_vec = np.zeros(shape=(1, wordvec_params['embedding_dim']))
    i = 0
    for word, count in word_frequency_list[-infreq_word_size:]:
        if word in word_vecs:
            i += 1
            infreq_word_vec += word_vecs[word]
    infreq_word_vec = infreq_word_vec / i
    print(infreq_word_vec)

    # build word matrix
    #word_matrix = np.zeros(shape=(vocab_size, wordvec_params['embedding_dim']))
    word_matrix = np.repeat(infreq_word_vec, vocab_size, axis=0)
    vocab_word_count = 0
    infreq_word_count = 0
    oov_count = 0
    zero_vector_count = 0
    infreq_with_coocrence_count = 0
    infreq_no_coocrence_count = 0
    infreq_words = []
    for i in range(1, vocab_size):
        if i % 10000 == 0:
            print("found {} in vocab words".format(vocab_word_count))
            print("found {} infreq words".format(infreq_word_count))
            print("found {} oov words".format(oov_count))
            print("found {} infreq with coocrence words".format(infreq_with_coocrence_count))
            print("found {} infreq without coocrence words".format(infreq_no_coocrence_count))
        # word in word vecs
        if inv_vocab[i] in word_vecs:
            vocab_word_count += 1
            word_matrix[i] = word_vecs[inv_vocab[i]] # 0 will be used as padding, so start from 1
        # use something to represent the unknown word
        else:
            oov_count += 1
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
                    infreq_words.append(inv_vocab[i])
                    infreq_word_count += 1
                else:
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
                        infreq_with_coocrence_count += 1
                        for i in range(4):
                            if context_avail[i] > 0:
                                # sum existing word vecs
                                context_vecs += word_vecs[inv_vocab[context_index[i]]]
                        # average the word vecs as this infrequent word's vec
                        word_matrix[i] = context_vecs / sum(context_avail)
                    # not enough context words, use averaged infrequent words
                    else:
                        infreq_no_coocrence_count += 1
                        infreq_words.append(inv_vocab[i])
                        word_matrix[i] = infreq_word_vec
    
    for i in range(vocab_size):
        if la.norm(word_matrix[i], ord=2) < 0.0000001:
            zero_vector_count += 1
            word_matrix[i] = infreq_word_vec
            print(inv_vocab[i])
            print(i)
    print(zero_vector_count)    
            
    with open(general_params['temp_home'] + 'infreq_words.txt', 'w') as outf:
        for word in infreq_words:
            outf.write(word + '\n')

    # save word matrix to file
    with open(general_params['temp_home'] + wordvec_params['wordmatrix_file'], 'wb') as outf:
        pickle.dump(word_matrix, outf)
    return word_matrix

# transform tweets (list of words) to ndarray of vectors
def transform_with_padding(tweets, vocab, max_len=20, direction='pre'):
    # Better looking solution with one line of code
    tweets_word_index = [[vocab[w] for w in t if w in vocab] for t in tweets]
    return pad_sequences(tweets_word_index, maxlen=max_len, padding=direction, truncating=direction)

# transfer tweet (in the form of vocab index) into mean of its embeddings of the words
# input should be padded to same length
def embedding_mean_vectorizer(tweets_vocab_index, word_matrix):
    vectors = []
    for i in range(tweets_vocab_index.shape[0]):
        vector = []
        index_list = tweets_vocab_index[i]
        for idx in index_list:
            if int(idx) == 0:
                continue
            if la.norm(word_matrix[int(idx)], ord=2) < 0.000001:
                print(idx)
                print(word_matrix[int(idx)])
            vector.append((1.0 / la.norm(word_matrix[int(idx)], ord=2)) * word_matrix[int(idx)])
        vectors.append(np.mean(vector, axis=0))
    return np.asarray(vectors)

# transfer tweet (in the form of vocab index) into concatenate of its word embeddings
# input should be padded to same length
def embedding_concat_vectorizer(tweets_vocab_index, word_matrix):
    vectors = []
    for i in range(tweets_vocab_index.shape[0]):
        vector = []
        index_list = tweets_vocab_index[i]
        for idx in index_list:
            vector.append(word_matrix[int(idx)])
        vectors.append(np.concatenate(vector))
    return np.asarray(vectors)