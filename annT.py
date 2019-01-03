# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 18:20:10 2018

@author: Tomasz
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import theano 
import tensorflow
import keras

dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:,[3,4,5,6,7,8,9,10,11,12]].values
y = dataset.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lb_1 = LabelEncoder()
lb_2 = LabelEncoder()
X[:,1] = lb_1.fit_transform(X[:,1])
X[:,2] = lb_2.fit_transform(X[:,1])
ohe = OneHotEncoder(categorical_features = [1])
X = ohe.fit_transform(X).toarray()
X = X[:, 1:]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2)

from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train,y_train, batch_size = 10, nb_epoch = 100)


y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)