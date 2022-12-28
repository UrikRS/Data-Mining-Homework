import pandas as pd
from sklearn.preprocessing import StandardScaler

def map_attributes(data):
    gender_map = {'Male':0, 'Female':1}
    country_map = {'France':0, 'Spain':1, 'Germany':2}
    data['gender'] = data['gender'].map(gender_map)
    data['country'] = data['country'].map(country_map)

def load_data(directory):
    data = pd.read_csv(directory+'/Bank_Customer.csv', sep=',', header=0)
    map_attributes(data)
    features = ['credit_score','country','gender','age','tenure','balance','products_number','credit_card','active_member','estimated_salary']
    continuous = ['credit_score','age','tenure','balance','products_number','estimated_salary']
    data[continuous] = StandardScaler().fit_transform(data[continuous])
    return data[features], data[['churn']]