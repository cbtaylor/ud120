# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 23:14:34 2016

@author: Brian
"""


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

corpus = ['This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?']
X = vectorizer.fit_transform(corpus)
print X  

from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")

print stemmer.stem("responsiveness")