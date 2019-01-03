# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 11:03:16 2018

@author: Tomokoko
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Mall_Customers.csv')

X = dataset.iloc[:,[3,4]].values


from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', n_init = 10, max_iter = 300, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)



kmeans = KMeans(n_clusters = 5, init = 'k-means++', n_init = 10, max_iter = 300, random_state = 0)
y_kmeans =  kmeans.fit_predict(X)
v = ['red','blue','green','orange','purple']
plt.xlim(X[:,0].min()-1,X[:,0].max()+1)
plt.ylim(X[:,1].min()-1,X[:,1].max()+1)
for j in range(0,5): 
    plt.scatter(X[y_kmeans == j,0],X[y_kmeans == j,1], c = v[j] )
    plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1])