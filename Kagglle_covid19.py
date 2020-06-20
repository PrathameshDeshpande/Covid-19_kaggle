import numpy as np
import pandas  as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn

dataset=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
dataset.loc[dataset['Province_State'].isnull(), 'Province_State'] = 'A'
dataset['Date']=pd.to_datetime(dataset.Date)
dataset['month']=dataset.Date.dt.month
dataset['dayofweek']=dataset.Date.dt.dayofweek
dataset['D']=dataset.Date.dt.day
cols = list(dataset.columns)
X_train=dataset[cols[1:3]+cols[6:9]].values
y_train=dataset.iloc[:,4:6].values

a=X_train.dtype
print(a)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
transformer = ColumnTransformer(transformers=[('P', OneHotEncoder(), [0, 1])])
X_train = transformer.fit_transform(X_train)
