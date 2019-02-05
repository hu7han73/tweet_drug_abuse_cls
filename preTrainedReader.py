# by huhan3

from __future__ import division, print_function, unicode_literals

import numpy as np
from numpy import linalg as la
import array

# This is a helper class for reading pre-trained word vector files
# Now supports 1) Word2Vec (binary file) 2) GloVe (text file)
# Word2Vec default parameter set for that GoogleNews file, change 'encoding' and 'newLine' parameter for normal file
# use it like a dictionary
class PreTrainedVectorReader(object):
    def __init__(self, vocab, vectors, vector_size):
        self.vocab = vocab
        self.vectors = vectors
        self.vector_size = vector_size

        self.dic = {}
        for i, word in enumerate(vocab):
            self.dic[word] = i

    def __str__(self):
        return str(self.vectors)

    def __len__(self):
        return len(self.dic)

    def __getitem__(self, word):
        return self.get_vector(word)

    def __contains__(self, word):
        return word in self.dic

    def get_vector(self, word):
        try:
            index = self.dic[word]
            return self.vectors[index]
        except KeyError:
            return None

    def get_word(self, index):
        try:
            return self.vocab[index]
        except IndexError:
            return None

    def get_index(self, word):
        try:
            return self.dic[word]
        except KeyError:
            return None

    def add_word(self, word, vector):
        if len(vector) != self.vector_size:
            print('New vector size = {}, existing vector size = {}'.format(str(len(vector)), str(self.vector_size)))
        else:
            if word in self.dic:
                print('Word already exists')
            else:
                self.dic[word] = vector

    @classmethod
    def from_word2vec_binary(cls, filename, vocabUnicodeSize=78, desired_vocab=None, encoding='ISO-8859-1', newLines=False, unitvector=False, decode_error='ignore'):
        """
        Parameter:
        -----------------------
        filename: The file name of the desired binary file
        vocabUnicodeSize: Maximum string length (78 default)
        desired_vocab: A list (set is better) of words.
            All words not in this list will be ignored.
            (Note this list can contain bytes or str)
        encoding: The codec used for the file
        newlines: If there's an empty char bofore a new line
        unitvector: If convert vectors into unitvectors

        Returns:
        -------------------
        The PreTrainedVectorReader object
        """
        with open(filename, 'rb') as inf:
            header = inf.readline() # read header and get size
            vocab_size, vector_size = list(map(int, header.split()))

            vocab = np.empty(vocab_size, dtype='<U%s' % vocabUnicodeSize) # init vocab (little-endian) and vectors
            vectors = np.empty((vocab_size, vector_size), dtype=np.float64)
            binary_len = np.dtype(np.float32).itemsize * vector_size # important to know how long a vector is
            for i in range(vocab_size):
                word = b''
                while True:
                    ch = inf.read(1)
                    if ch == b' ':
                        break
                    word += ch
                include = desired_vocab is None or word.decode(encoding, errors=decode_error) in desired_vocab or word in desired_vocab # check if need to ignore this word
                if include:
                    vocab[i] = word.decode(encoding, errors=decode_error)

                vector = np.fromstring(inf.read(binary_len), dtype=np.float32) # read bytes and convert to vector
                if include:
                    if unitvector:
                        vectors[i] = (1.0 / la.norm(vector, ord=2)) * vector # convert this vector to unitvector
                    else:
                        vectors[i] = vector
                if newLines:
                    inf.read(1) # for normal file, read a empty char to begin a newline, not needed for GoogleNews file

            if desired_vocab is not None:
                vectors = vectors[vocab != '', :] # this is numpy's vector operation, find out all index of not-empty strings
                vocab = vocab[vocab != '']

        return cls(vocab=vocab, vectors=vectors, vector_size=vector_size)

    @classmethod
    def from_glove_plain(cls, filename, vocab_size=500000, vocabUnicodeSize=78, desired_vocab=None, encoding='utf-8', unitvector=False):
        """
        Parameter:
        -----------------------
        filename: The file name of the desired plaintext file
        vocab_size: The maximun number of vocab size, 5000000 default
        vocabUnicodeSize: Maximum string length (78 default)
        desired_vocab: A list (set is better) of words.
            All words not in this list will be ignored.
            (Note this list can contain bytes or str)
        encoding: The codec used for the file
        newlines: If there's an empty char bofore a new line
        unitvector: If convert vectors into unitvectors

        Returns:
        -------------------
        The PreTrainedVectorReader object
        """

        init = False
        vector_size = 0
        c = 0
        with open(filename, 'rt', encoding=encoding) as inf:
            for i, line in enumerate(inf):
                raw = line.split(' ')

                if not init:
                    if desired_vocab:
                        vocab_size = len(desired_vocab)
                    vector_size = len(raw) - 1
                    vocab = np.empty(vocab_size, dtype='<U%s' % vocabUnicodeSize) # init vocab (little-endian) and vectors
                    vectors = np.empty((vocab_size, vector_size), dtype=np.float64)
                    init = True

                vector = np.array([float(x) for x in raw[1:]], dtype=np.float64)
                if desired_vocab is None or raw[0] in desired_vocab:
                    vocab[c] = raw[0]
                    if unitvector:
                        vectors[c] = (1.0 / la.norm(vector, ord=2)) * vector # convert this vector to unitvector
                    else:
                        vectors[c] = vector
                    c += 1

        vectors = np.resize(vectors, (c,vector_size)) # resize to remove empty elements
        vocab = np.resize(vocab, c)

        return cls(vocab=vocab, vectors=vectors, vector_size=vector_size)

# some example of usage:
# from pre_trained_reader import PreTrainedVectorReader as ptvr
# for GoogleNews
# vectors = ptvr.from_word2vec_binary(name, desired_vocab=vocab)
# for glove
# vectors = ptvr.from_glove_plain(name, desired_vocab=vocab)
# for Fréderic Godin dataset and DSM (A corpus for mining drug-related knowledge from Twitter chatter: Language models and their utilities)
# vectors = ptvr.from_word2vec_binary(name, encoding='utf8', newLines=True, desired_vocab=vocab)