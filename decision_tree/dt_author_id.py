#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import tree
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### create classifier
clf = tree.DecisionTreeClassifier(min_samples_split=40)

### fit the classifier features and labels
t0 = time()
clf.fit(features_train, labels_train)  
print "training time:", round(time()-t0, 3), "s"

### use the trained classifier to predict labels for the test features
t0 = time()
pred = clf.predict(features_test)
print "predicting time:", round(time()-t0, 3), "s"

### calculate and return the accuracy on the test data
acc = accuracy_score(labels_test, pred)
print
print "Accuracy:", acc
accuracy = clf.score(features_test, labels_test)
print "Accuracy:", accuracy
print
print sum(pred)

from sklearn.metrics import classification_report
target_names = ['Sara', 'Chris']
print(classification_report(labels_test, pred, target_names=target_names))
#########################################################


