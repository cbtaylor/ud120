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
from data_prep import data_prep

labels, features = data_prep()

print type(labels)
print type(features)

"""
p = gg.ggplot(aes(x='salary', y='bonus', color='poi'), data = df) + \
    gg.geom_point() + \
    gg.scale_x_continuous(limits = [0,1200000]) + \
    gg.scale_y_continuous(limits = [0,8500000]) + \
    gg.geom_smooth(method='lm') 
print p

q = gg.ggplot(aes(x='salary', y='long_term_incentive', color='poi'), data = df) + \
    gg.geom_point() + \
    gg.scale_x_continuous(limits = [0,1200000]) + \
    gg.scale_y_continuous(limits = [0,5200000]) + \
    gg.geom_smooth(method='lm') 
print q

r = gg.ggplot(aes(x='total_stock_value', y='exercised_stock_options', \
    color='poi'), data = df) + \
    gg.geom_point() + \
    gg.scale_x_log10() + \
    gg.scale_y_log10() + \
    gg.xlim(100000,50000000)
print r
"""

from sklearn import datasets
iris = datasets.load_iris()
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(features, labels).predict(features)
print("Number of mislabeled points out of a total %d points : %d"
      % (features.shape[0],(labels != y_pred).sum()))
