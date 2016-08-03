#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### it's all yours from here forward!  

from sklearn.tree import DecisionTreeClassifier
from time import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import numpy as np
from sklearn import cross_validation


print len(features), len(labels)

features_train, features_test, labels_train, labels_test = \
    cross_validation.train_test_split(features, labels, \
    test_size=0.3, random_state=42)

### create classifier
clf = DecisionTreeClassifier()


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
print "Accuracy:", acc
accuracy = clf.score(features_test, labels_test)
print "Accuracy:", accuracy

print



target_names = ['non-POI', 'POI']
print(classification_report(labels_test, pred, target_names = target_names))
