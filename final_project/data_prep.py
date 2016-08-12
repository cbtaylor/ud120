#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

import pandas as pd

def data_prep():
    ### Load the dictionary containing the dataset
    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)
    
    ### Task 2: Remove outliers
    del data_dict['TOTAL']
    del data_dict['THE TRAVEL AGENCY IN THE PARK']

    
    #pprint.pprint(data_dict)
    df = pd.DataFrame.from_dict(data_dict, orient='index', dtype=None)
    
    
    #df.loc[df.salary == 'NaN', 'salary'] = np.nan
    
    df['salary'] = \
        pd.to_numeric(df.salary, errors='coerce')
    
    df['bonus'] = \
        pd.to_numeric(df.bonus, errors='coerce')
    
    df['to_messages'] = \
        pd.to_numeric(df.to_messages, errors='coerce')
    
    df['deferral_payments'] = \
        pd.to_numeric(df.deferral_payments, errors='coerce')
    
    df['exercised_stock_options'] = \
        pd.to_numeric(df.exercised_stock_options, errors='coerce')
    
    df['restricted_stock'] = \
        pd.to_numeric(df.restricted_stock, errors='coerce')
    
    df['shared_receipt_with_poi'] = \
        pd.to_numeric(df.shared_receipt_with_poi, errors='coerce')
    
    df['restricted_stock_deferred'] = \
        pd.to_numeric(df.restricted_stock_deferred, errors='coerce')
    
    df['total_stock_value'] = \
        pd.to_numeric(df.total_stock_value, errors='coerce')
    
    df['expenses'] = \
        pd.to_numeric(df.expenses, errors='coerce')
    
    df['loan_advances'] = \
        pd.to_numeric(df.loan_advances, errors='coerce')
    
    df['from_messages'] = \
        pd.to_numeric(df.from_messages, errors='coerce')
    
    df['other'] = \
        pd.to_numeric(df.other, errors='coerce')
    
    df['from_this_person_to_poi'] = \
        pd.to_numeric(df.from_this_person_to_poi, errors='coerce')
    
    df['director_fees'] = \
        pd.to_numeric(df.director_fees, errors='coerce')
    
    df['deferred_income'] = \
        pd.to_numeric(df.deferred_income, errors='coerce')
    
    df['long_term_incentive'] = \
        pd.to_numeric(df.long_term_incentive, errors='coerce')
    
    df['from_poi_to_this_person'] = \
        pd.to_numeric(df.from_poi_to_this_person, errors='coerce')
    
    
    
    # after looking at the descriptions and the corrlation matrix
    # I can see what to get rid of
    
    # Eliminate columns with too many missing values, and the email
    # column, which can't be used
    # Also drop from_poi and to_poi, which I don't feel comfortable using
    # If the classifier is supposed to identify a poi then it seems
    # inappropiate to use to_poi and from_poi. This seems like leakage.
    # It's a bit like the example of trying to identify cancer patients
    # from an MRI, but using information about what facility they're at.
    # If they're already at a cancer treatment facility then the 
    # classifier isn't worth much.
    df = df.drop(['deferral_payments', 
                  'loan_advances', 
                  'director_fees', 
                  'from_this_person_to_poi',  
                  'from_poi_to_this_person',
                  'shared_receipt_with_poi', 
                  'email_address'], axis = 1)
    
    # combine restricted_stock and restricted_stock_deferred
    # because they must be closely related and there are only 17
    # values for restricted_stock_deferred
    df['restricted_stock'] = df.restricted_stock.fillna(0)
    df['restricted_stock_deferred'] = df.restricted_stock_deferred.fillna(0)
    df['restricted_stock_combined'] = \
        df.restricted_stock + df.restricted_stock_deferred
    df = df.drop(['restricted_stock', 'restricted_stock_deferred'], axis=1)
    
    labels = df.poi
    features = df.drop('poi', axis=1)
    
    return labels, features