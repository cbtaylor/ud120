#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import pprint
import pandas as pd
import ggplot as gg
from ggplot import aes
import matplotlib.pyplot as plt
import numpy as np

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
all_features =  ['poi',                       #0
                 'salary',                    #1
                 'exercised_stock_options',   #2
                 'expenses',                  #3
                 'bonus',                     #4
                 'restricted_stock',          #5
                 'deferral_payments',         #6
                 'total_payments',            #7
                 'other',                     #8
                 'director_fees',             #9
                 'deferred_income',          #10
                 'long_term_incentive',      #11
                 'restricted_stock',         #12
                 'restricted_stock_deferred',#13
                 'total_stock_value',        #14
                 'loan_advances',            #15
                 'to_messages',              #16
                 'from_messages',            #17
                 'shared_receipt_with_poi',  #18
                 'from_this_person_to_poi',  #19
                 'from_poi_to_this_person'   #20
                 ] 

features_list = [all_features[0],
                 all_features[1],
                 all_features[4],
                 all_features[16],
                 all_features[17],
                 all_features[14]]

print features_list               
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
# remove Total and Travel AGency


#pprint.pprint(data_dict)
df = pd.DataFrame.from_dict(data_dict, orient='index', dtype=None)
print df.describe()
print df.head()

print len(data_dict)
del data_dict['TOTAL']
del data_dict['THE TRAVEL AGENCY IN THE PARK']
print len(data_dict)

p = gg.ggplot(aes(x='salary', y='bonus'), data = df)
print p + gg.geom_point()



### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

from sklearn import tree
from sklearn import grid_search

parameters = {'max_depth':[2,3,4],
              'max_features':[2,3,4],
              'min_samples_leaf':[2,3,4]}
dtree = tree.DecisionTreeClassifier()
clf = grid_search.GridSearchCV(dtree, parameters)


from sklearn import svm
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svr = svm.SVC()
clf = grid_search.GridSearchCV(svr, parameters)



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

clf.fit(features_train, labels_train)

 

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)