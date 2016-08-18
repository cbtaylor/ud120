#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

import pandas as pd

from data_prep import data_prep
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# fetch the data
y, X = data_prep()

print "labels:", y.shape
print "features:", X.shape

# using k=8 generated the best F1 score
X_new = SelectKBest(chi2, k=8).fit_transform(X, y)
print "after select k best:", X_new.shape




gnb = GaussianNB()
num_folds = 4
kf = StratifiedKFold(y, n_folds = num_folds, 
                     shuffle = True, random_state = 19)
agg_cm = ([[0,0],[0,0]])

for train_indices, valid_indices in kf:
    X_train, y_train = X_new[train_indices], y[train_indices]
    X_valid, y_valid = X_new[valid_indices], y[valid_indices]
    
    y_pred = gnb.fit(X_train, y_train).predict(X_valid)
    
    target_names = ['not a poi', 'poi']
    print(classification_report(y_valid, y_pred, target_names=target_names))
    
    cm = confusion_matrix(y_valid, y_pred, labels = [False, True])
    print cm
    agg_cm += cm

print "Aggregate confusion matrix:\n", agg_cm
    
true_neg = agg_cm[0][0]
false_pos = agg_cm[0][1]
false_neg = agg_cm[1][0]
true_pos = agg_cm[1][1]

precision = float(true_pos) / (true_pos + false_pos)
recall = float(true_pos) / (true_pos + false_neg)
f1 = 2 * precision * recall / (precision + recall)

print
print " true positives:", true_pos
print " true negatives:", true_neg
print "false positives:", false_pos
print "false negatives:", false_neg
print
print "precision:", precision
print "   recall:", recall
print " F1 score:", f1