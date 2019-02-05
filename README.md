# tweet_drug_abuse_cls
Demo code for classifing drug abuse tweets using ensembled CNN

This repo is requested as a demo project for learning deep learning. It is derived from the code for an academic projects of mine.

I simplified the original code (it was and is a big mess) I used so it will be easier to run and read. However, this version is not tested and may (certainly) contain bugs. Please try to debug yourself, but I can help when necessary.

The object of the project is to train a model (or, actually, the ensemble of a collection of models) that can classify tweets as drug-abuse related or non drug-abuse related. We tried to compare the performance between traditional machine learning methods, such as SVM, RandomForest, and NaiveBayes, and CNN-based methods, including word-level CNN and char-level CNN. The project also explores how dataset imbalanceness (as it would be in real world, more negative labels than positive labels) impacts model performances. The traditional ML methods are described in this paper: "Social Media Mining for Toxicovigilance: Automatic Monitoring of Prescription Medication Abuse from Twitter." The CNN models are described in the paper "cnn_models_paper.pdf", which you can find in the repo.

#### What you need:
  ##### Environments:
    Python 3 (I use python 3.5)
    Java
    Nvidia GPU
  ##### Python packages:
    tensorflow-gpu
    keras
    sklearn
    numpy
    pandas
    nltk
    scipy
    gensim
    

#### Here we describe each component of the repo:
##### Datset (/data):
  * "MT_5k_labels_raw_tweets_clean_tweets_unique.csv": In the /data folder we have a csv file contains several fields. including: [index]; [MT_labels] Labels made by MechanicalTurk workers; [clean_tweets] Preprocessed tweets; [final_label] Actual label of the tweet, volted from the MT_labels; [raw_tweet] Raw tweet content, collected from Twitter. There are about 4,985 tweets, with almost balanced number of positive and negative labels.
  * "custom_stopwords2.txt": A list of stopwords I used. I removed some potentially useful words from the commonly used stopword list.
  * "word_clusters.txt": File contains word cluster information as described in that paper.
  
##### Preprocessor (/preprocessor):
  A handy tweet preprocessor copied from https://github.com/s/preprocessor.
  
##### results (/results):
  Folder where results can go.
  
##### temp (/temp):
  Temp files can go there.
  
##### word_vectors (/word_vectors):
  You can download pre-trained word vectors from these links (each is weighted at several Gigs, prepare some disk space :):
  
  * word2vec trained on GoogleNews dataset: https://www.dropbox.com/s/y5vhbe53d4w0wuk/GoogleNews-vectors-negative300.bin?dl=0 (source:https://github.com/mmihaltz/word2vec-GoogleNews-vectors).
  
  * word2vec trained on tweets by Godin: https://www.dropbox.com/s/1urozhtu3wya3m6/Godin_word2vec_twitter_model.bin?dl=0 (source: https://fredericgodin.com/software/).
  
  * glove trained on tweets: https://www.dropbox.com/s/f40nydv41sn5a8k/glove.42B.300d.txt?dl=0 (source: https://nlp.stanford.edu/projects/glove/).
  
  * word2vec trained on drug related tweets: https://www.dropbox.com/s/zfqzr6s2n809aqw/DSM_word2vec_model.bin?dl=0 (source: https://www.sciencedirect.com/science/article/pii/S2352340916307168).
    
##### Core files (/):
  * "ark-tweet-nlp-0.3.2.jar": The trained tweet POS-tagger, check http://www.cs.cmu.edu/~ark/TweetNLP/.
  * "CMUTweetTagger.py": Python wrapper of the POS tagger.
  * "runTagger.sh": Used by POS tagger.
  * "ensemble_ml_models.py": Train machine learning models, make predictions on validation and testing dataset.
  * "ensemble_models.py": Train CNN models, make predictions on validation and testing dataset.
  * "feature_expansion.py": Perform feature expansion.
  * "LSTM_CONV.py": Build tensorflow models (CNN, LSTM, etc). Including a lot of models, but only some of them are actually used.
  * "NB_SVM.py": Build ML models.
  * "preTrainedReader.py": Helper library to read pre-trained word vectors.
  * "read_ml_results.py": After ML models are trained, use this to ensemble and evaluate the results.
  * "read_results.py": After CNN models are trained, use this to ensemble and evaluate the results.
  * "TemporalMeanPooling.py": A component of some LSTM models, not actually used but is here for dependency.
  * "training_params.py": Common parameters for the models and the training process.
  * "training_utils.py": Some helper functions for reading data, preprocessing data, etc.

#### How to run:
  * First setup all the environments, and clone the repo. 
  * Read all the related papers and the code so you understand the big picture.
  * If you want to try some other preprocess methods, you can read the raw tweets and write your own code to preprocess them. Otherwise, you can use clean tweets in the dataset.
  * Run the "feature_expansion.py" to get the expanded features ready.
  * Run the "ensemble_models.py" and the "ensemble_ml_models.py" to train models.
  * Run the "read_results.py" and the "read_ml_results.py" to evaluate the models.
  * Do anything you think is interesting.
  
#### Again, there most likely will be bugs in the code. Don't expect everything works on first try.

#### Good Luck!
  
  
