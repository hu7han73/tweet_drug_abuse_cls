import pandas as pd
import numpy as np
#from keras.utils import to_categorical
import itertools
import pandas as pd
import numpy as np
import preprocessor as p
import re
from nltk.stem.porter import PorterStemmer

#===========================================================
# pre-process (without stemming) a raw tweet and return a list of words
def raw_tweet_prep(raw_tweet, stopwords, space_replace_re, repeating_re, single_char_re):
    #raw_tweet = ' '.join(raw_tweet)
    tweet_tokenized = p.tokenize(raw_tweet.lower().replace('\n',' '))
    tweet_tokenized = space_replace_re.sub(' ', tweet_tokenized)
    tweet_tokenized = repeating_re.sub(r"\1", tweet_tokenized)
    #tweet_tokenized = single_char_re.sub(' ', tweet_tokenized)
    tweet_tokenized = tweet_tokenized.strip().split()
    words = [w for w in tweet_tokenized if w not in stopwords]
    if len(words) > 1:
        return words
    else:
        raise Exception("Input tweet too short")

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
def raw_tweet_prep_stem(raw_tweet, stopwords, stemmer, space_replace_re, repeating_re, single_char_re):
    # tokenize and replace url with 'URL', numbers with 'NUMBER' and 'EMOJi'
    tweet_tokenized = p.tokenize(raw_tweet.lower().replace('\n',' '))
    # remove some more things ('s, 'm, 't, html symbol, other non english char, and repeating expression)
    tweet_tokenized = space_replace_re.sub(' ', tweet_tokenized)
    tweet_tokenized = repeating_re.sub(r"\1", tweet_tokenized)
    #tweet_tokenized = single_char_re.sub(' ', tweet_tokenized)
    tweet_tokenized = tweet_tokenized.strip().split()
    words = [stemmer.stem(w) for w in tweet_tokenized if w not in stopwords]
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

# compile the regex patterns neededfor pre-processing
def compile_re2():
    html_entities_re = r"&#?\w+;"
    quote_s_re = "'s|'t|'m"
    non_char_re = "[^ 0-9a-zA-Z]"
    html_compiled = re.compile(html_entities_re)
    space_replace_compiled = re.compile('|'.join([quote_s_re, non_char_re]))
    single_char_compiled = re.compile(r"(?:\b\w\b)+")
    repeating_compiled = re.compile(r"([a-zA-Z])\1\1+")
    return html_compiled, space_replace_compiled, repeating_compiled, single_char_compiled
