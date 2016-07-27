#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from sklearn import neighbors
from sklearn.metrics import classification_report
from time import time

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

for n_neighbors in [1,2,3,4,5,7,9,11,13,15,17,19,21,23,25,35,50]:
    for weights in ['uniform', 'distance']:
        print "k =", n_neighbors, "weights =", weights
        
        ### create classifier
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
        
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
        
        
        target_names = ['fast', 'slow']
        print(classification_report(labels_test, pred, target_names=target_names))

### draw the pic for k=3
n_neighbors, weights = (50, "distance")
clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass






