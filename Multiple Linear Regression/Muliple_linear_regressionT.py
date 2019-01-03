# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 23:47:39 2018

@author: Tomokoko
"""

import numpy as np
import matplotlib.pyplot as mtl
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3]) 
encoder = OneHotEncoder(categorical_features = [3])
X = encoder.fit_transform(X).toarray()

#Avoiding the Dummy Variable Trap 
X = X[:, 1:]

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size = 0.8, random_state = 0)

from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(X_train,y_train)


y_pred = regressor.predict(X_test)


import statsmodels.formula.api as nm

X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_optimal = X[:,[0,1,2,3,4,5]]
regressor_OLS = nm.OLS(endog = y, exog = X_optimal).fit()
regressor_OLS.summary()

