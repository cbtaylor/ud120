#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    import pandas as pd
    
    df = pd.DataFrame(ages, columns = ['age'])
    df['worth'] = net_worths
    df['preds'] = predictions
    df['error'] = (df.preds - df.worth) ** 2
    
    cutoff = df.error.quantile(0.9)
    
    cleaned = df[df.error <= cutoff]
    
    for i in range(cleaned.shape[0]):
        tp = (cleaned.age.iloc[i], cleaned.worth.iloc[i], cleaned.error.iloc[i])
        cleaned_data.append(tp)
    
    return cleaned_data

