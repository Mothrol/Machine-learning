# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 15:18:09 2018

@author: Tomokoko
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values
"""from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_optimall,y,train_size = 0.8, random_state = 0)"""

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

from sklearn.preprocessing import PolynomialFeatures 
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X,y, color = 'red')
plt.plot(X,lin_reg.predict(X), color = 'b')
plt.plot(X_grid,lin_reg_2.predict(poly_reg.fit_transform(X_grid)))
plt.show()

lin_reg_2.predict(poly_reg.fit_transform(6.5))