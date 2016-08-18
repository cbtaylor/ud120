#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

import pandas as pd

from data_prep import data_prep
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from math import log

# fetch the data
y, X = data_prep()

print "labels:", y.shape
print "features:", X.shape


# total_stock_value is the single best feature

X_new = SelectKBest(chi2, k=1).fit_transform(X, y)
print "after select k best:", X_new.shape

knn = KNeighborsClassifier(n_neighbors=4, weights='distance', 
                           algorithm='auto', leaf_size=30, p=2, 
                           metric='minkowski', metric_params=None, n_jobs=1)

          
num_folds = 4
kf = StratifiedKFold(y, n_folds = num_folds, 
                     shuffle = True, random_state = 19)
agg_cm = ([[0,0],[0,0]])

for train_indices, valid_indices in kf:
    X_train, y_train = X_new[train_indices], y[train_indices]
    X_valid, y_valid = X_new[valid_indices], y[valid_indices]
    
    y_pred = knn.fit(X_train, y_train).predict(X_valid)
    
    target_names = ['not a poi', 'poi']
    print(classification_report(y_valid, y_pred, target_names=target_names))
    
    cm = confusion_matrix(y_valid, y_pred, labels = [False, True])
    print cm
    agg_cm += cm
    print "     MCC:", matthews_corrcoef(y_valid, y_pred)
    print

print "\nAggregate confusion matrix:\n", agg_cm
    
TN = agg_cm[0][0]
FP = agg_cm[0][1]
FN = agg_cm[1][0]
TP = agg_cm[1][1]

try:
    precision = float(TP) / (TP + FP)
    recall = float(TP) / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    mcc = (TP * TN - FP * FN) / \
          ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5 
    
    print
    print " true positives:", true_pos
    print " true negatives:", true_neg
    print "false positives:", false_pos
    print "false negatives:", false_neg
    print
    print "precision:", precision
    print "   recall:", recall
    print " F1 score:", f1
    print "      MCC:", mcc

except:
    print
    print "Scores can't be calculated because there are no positive predictions"