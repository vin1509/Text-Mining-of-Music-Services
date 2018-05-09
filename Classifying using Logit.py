# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 23:31:25 2017

@author: XtremeUser
"""

import pandas as pd
import os


os.chdir('C:/Users/XtremeUser/Google Drive/MIS Project/Rescraped Files/Classifier Sets')

train_data_df = pd.read_csv('data_full_train.csv')
test_data_df = pd.read_csv('data_clean_test_all.csv')
train_data_df = pd.DataFrame.dropna(train_data_df)

import re, nltk
from sklearn.feature_extraction.text import CountVectorizer        
from nltk.stem.porter import PorterStemmer

#######
# based on http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html
stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    # remove non letters
    text = re.sub("[^a-zA-Z]", " ", text)
    # tokenize
    tokens = nltk.word_tokenize(text)
    # stem
    stems = stem_tokens(tokens, stemmer)
    return stems
######## 

vectorizer = CountVectorizer(
    analyzer = 'word',
    tokenizer = tokenize,
    lowercase = True,
    stop_words = 'english',
    max_features = 85
)

corpus_data_features = vectorizer.fit_transform(train_data_df.Text.tolist() + test_data_df.Text.tolist())
corpus_data_features_nd = corpus_data_features.toarray()

vocab = vectorizer.get_feature_names() 

import numpy as np
# Sum up the counts of each vocabulary word
dist = np.sum(corpus_data_features_nd, axis=0)

# For each, print the vocabulary word and the number of times it 
# appears in the training set
for tag, count in zip(vocab, dist):
     print(count, tag)
     
from sklearn.cross_validation import train_test_split

# remember that corpus_data_features_nd contains all of our original train and test data, so we need to exclude
# the unlabeled test entries
X_train, X_test, y_train, y_test  = train_test_split(
    corpus_data_features_nd[0:len(train_data_df)], 
    train_data_df.Sentiment,
    train_size=0.85, 
    random_state=1234)

from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression()
log_model = log_model.fit(X=X_train, y=y_train)

from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression()
log_model = log_model.fit(X=X_train, y=y_train)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

# train classifier
log_model = LogisticRegression()
log_model = log_model.fit(X=corpus_data_features_nd[0:len(train_data_df)], y=train_data_df.Sentiment)

# get predictions
test_pred = log_model.predict(corpus_data_features_nd[len(train_data_df):])


# sample some of them
import random
spl = random.sample(range(len(test_pred)), 15)

# print text and labels
for text, sentiment in zip(test_data_df.Text[spl], test_pred[spl]):
    print (sentiment, text)