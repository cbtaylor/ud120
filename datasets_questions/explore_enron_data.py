#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import pprint

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

no_tp = 0
email = 0
poi = 0

for key in enron_data.keys():
    if enron_data[key]['poi']:
        poi += 1
    print enron_data[key]['total_payments']
    if enron_data[key]['total_payments'] == 'NaN':
        no_tp += 1


print 


print "number of poi:", poi
print "number of pois missing tp:", no_tp

"""
for key in enron_data.keys():
    print key

print


pprint.pprint(enron_data['SKILLING JEFFREY K'])
pprint.pprint(enron_data['FASTOW ANDREW S'])
pprint.pprint(enron_data['LAY KENNETH L'])
"""