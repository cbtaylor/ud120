#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

### use a small subset of the data
#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 


#########################################################
### import the sklearn module for SVM
from sklearn import svm

for C in [10000]:
    ### create classifier
    clf = svm.SVC(kernel = "rbf", C = C)
    
    ### fit the classifier features and labels
    t0 = time()
    clf.fit(features_train, labels_train)  
    print "training time:", round(time()-t0, 3), "s"
    
    ### use the trained classifier to predict labels for the test features
    t0 = time()
    pred = clf.predict(features_test)
    print "predicting time:", round(time()-t0, 3), "s"
    
    for i in [10, 26, 50]:
        print pred[i]
    ### calculate and return the accuracy on the test data
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(labels_test, pred)
    print
    print "C =", C, "Accuracy:", acc
    accuracy = clf.score(features_test, labels_test)
    print "C =", C, "Accuracy:", accuracy
    print
    print sum(pred)
    
    from sklearn.metrics import classification_report
    target_names = ['Sara', 'Chris']
    print(classification_report(labels_test, pred, target_names=target_names))
    
    

#########################################################


