# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 11:57:55 2017

@author: XtremeUser
"""

####Reviews Test
import nltk
import pandas as pd
import random
from collections import Counter


##Prep the Data



documents_am = pd.read_csv('Amazon_train.csv')

documents_am = pd.DataFrame.dropna(documents_am)

documents_am['Sentiment'] = documents_am['Sentiment'].map({1 : 'pos', 0 : 'neg'})

documents_am = documents_am[documents_am.columns[::-1]]

documents_am['Text'] = documents_am.apply(lambda row: nltk.word_tokenize(row['Text']), axis=1)

documents_list = documents_am.values.tolist()

documents_tuple = [tuple(l) for l in documents_list]

random.shuffle(documents_tuple)

##Get top words for the feature list

import itertools

am_no_doc1 = list(itertools.chain.from_iterable(documents_am['Text']))

#Create a Counter object to get the frequencies 

top_all = Counter(am_no_doc1)


all_words = nltk.FreqDist(w.lower() for w in am_no_doc1)
word_features = list(all_words)[:2000] 

##Feature function uses the top 2000 words to build a feature list

def document_features(document): 
    document_words = set(document) 
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

##Create feature list using the above function, create testing and training sets

featuresets = [(document_features(d), c) for (d,c) in documents_tuple]
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)
nltk.classify.accuracy(classifier, test_set)
classifier.show_most_informative_features(5)

    
classifier.classify(document_features('uninstall, crash, message, impossible'))

