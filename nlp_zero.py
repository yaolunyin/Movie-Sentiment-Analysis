#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 20:24:20 2022

@author: Yaolun Yin
"""


# Load and prepare the dataset
import nltk
from nltk.corpus import movie_reviews
import random
import numpy as np
import pandas as pd
#from sklearn.model_selection import KFold
import string
from nltk.stem import PorterStemmer
#from sklearn.model_selection import train_test_split
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB


# nltk.download('movie_samples')
# nltk.download('stopwords')
# nltk.download('wordnet')

# input
documents = [(list(movie_reviews.words(fileid)), category)
              for category in movie_reviews.categories()
              for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

# dataframe of documents
documents_df = pd.DataFrame(documents, columns=['review', 'sentiment'])



# However, a disadvantage of this method is that 
#  word_features contains lots of punctuation, which are 
#  irrelevant to sentiment analysis. Next, we want to improve method 1 by 
#   adding 3 steps:
#     1. removing punctuations
#     2. removing stop words, which do not affect the prediction result
#     3(not executed). For movie reviews, we can also remove actor names. (a good idea but I don't find a good package for it. 
#        Maybe code it manually later)
#     3. lemmatization of words. In laymen's term, we want to find the root of every word

# step 1: punctuation
def remove_punctuation(text):
    pun_lst = string.punctuation
    no_punct = [words for words in text if words not in pun_lst]
    #words_wo_punct=''.join(no_punct)
    return no_punct

documents_df['review'] = documents_df['review'].apply(lambda x: remove_punctuation(x))

# step 2: stopwords
stopword = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text

documents_df['review'] = documents_df['review'].apply(lambda x: remove_stopwords(x))

# step 2.1: numbers
def remove_numbers(text):
    text = [word for word in text if not word.isnumeric()]
    return text

documents_df['review'] = documents_df['review'].apply(lambda x: remove_numbers(x))

# step 3: stemming
ps =PorterStemmer()
def stemming(text):
    text = list(map(lambda x: ps.stem(x), text))
    # after stemming, remove duplicates
    text = list(np.unique(text))
    return text

documents_df['review'] = documents_df['review'].apply(lambda x: stemming(x))

# encode
def encode(cat):
    if cat == 1: 
        return 1
    else:
        return 0
documents_df['sentiment'] = documents_df['sentiment'].apply(lambda x: encode(x))

# turn dataframe into a list of tuples
documents_lst = [tuple(r) for r in documents_df.to_numpy()]

# # the rank of frequencies of all words
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
# # we call the 2000 most frequent words 'features'
word_features = list(all_words)[:2000]

# Define the feature extractor. 
# The input is a 'document', a list of words. The output is a dictionary that records whether a word exists in the 'doccument'.
def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['has({})'.format(word)] = (word in document_words)
    return features



labelsets = [(document_features(d), c) for (d,c) in documents_lst]
# train test split
training_set = labelsets[:1500]
testing_set = labelsets[1500:]
# classifier
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MultinomialNB accuracy percent:",nltk.classify.accuracy(MNB_classifier, testing_set))



# # cross-validation
# cv = KFold(n_splits=10,shuffle = True, random_state=1)
# clf =nltk.NaiveBayesClassifier
# print("Method 1: ----------------)
# for train_index, test_index in cv.split(labelsets):
#     train, test = labelsets[train_index], labelsets[test_index]
#     clf = nltk.NaiveBayesClassifier.train(train)
#     print('accuracy:', nltk.classify.util.accuracy(clf, test))
# # Result: average 0.8, not bad

# # Select top influential factors
# clf.show_most_informative_features(10)


