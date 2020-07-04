import numpy as np
import pandas  as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

dataset=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
dataset.loc[dataset['Province_State'].isnull(), 'Province_State'] = 'A'
dataset['Date']=pd.to_datetime(dataset.Date)
dataset['month']=dataset.Date.dt.month
dataset['dayofweek']=dataset.Date.dt.dayofweek
dataset['D']=dataset.Date.dt.day
cols = list(dataset.columns)
X_train=dataset[cols[1:3]+cols[7:9]].values
y_train=dataset.iloc[:,4:6].values
###################################################
test.loc[test['Province_State'].isnull(), 'Province_State'] = 'A'
test['Date']=pd.to_datetime(test.Date)
test['month']=test.Date.dt.month
test['dayofweek']=test.Date.dt.dayofweek
test['D']=test.Date.dt.day

col = list(test.columns)
test=test[col[1:3]+col[5:7]].values


labelencoder_X = LabelEncoder()
X_train[:, 0] = labelencoder_X.fit_transform(X_train[:, 0])
X_train[:, 1] = labelencoder_X.fit_transform(X_train[:, 1])
columnTransformer = ColumnTransformer(
            [('p', OneHotEncoder(), [1])],
            remainder='passthrough')
X_train = columnTransformer.fit_transform(X_train)
X_train = pd.DataFrame(X_train.toarray())
X_train = X_train.iloc[:,1:]
columnTransformer = ColumnTransformer(
            [('q', OneHotEncoder(), [183])],
            remainder='passthrough')
X_train = columnTransformer.fit_transform(X_train)
X_train = X_train[:,1:]
columnTransformer = ColumnTransformer(
            [('r', OneHotEncoder(), [316])],
            remainder='passthrough')
X_train = columnTransformer.fit_transform(X_train)
X_train = X_train[:,1:]

test[:, 0] = labelencoder_X.fit_transform(test[:, 0])
test[:, 1] = labelencoder_X.fit_transform(test[:, 1])
columnTransformer = ColumnTransformer(
            [('p', OneHotEncoder(), [1])],
            remainder='passthrough')
test = columnTransformer.fit_transform(test)
test = pd.DataFrame(test.toarray())
test = test.iloc[:,1:]
columnTransformer = ColumnTransformer(
            [('q', OneHotEncoder(), [183])],
            remainder='passthrough')
test= columnTransformer.fit_transform(test)
test = test[:,1:]
columnTransformer = ColumnTransformer(
            [('r', OneHotEncoder(), [316])],
            remainder='passthrough')
test = columnTransformer.fit_transform(test)
test = test[:,1:]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
test = sc.fit_transform(test)


model = tf.keras.models.Sequential([
  tf.keras.layers.Input(shape=(323,)),
  tf.keras.layers.Dense(400,activation ='relu'),
  tf.keras.layers.Dense(400,activation ='relu'),
    tf.keras.layers.Dense(400,activation ='relu'),
  tf.keras.layers.Dense(400,activation ='relu'),
tf.keras.layers.Dense(2,activation='sigmoid')
])

model.compile(optimizer='adam', loss='mse',metrics=['accuracy'])

r = model.fit(X_train, y_train,batch_size=64, epochs=2)


y_pred=model.predict(test)
y_pred = np.round(y_pred)

y_pred[0] = pd.Series(y_pred[0],name="Label")


