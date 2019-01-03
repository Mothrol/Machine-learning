# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 16:20:35 2018

@author: Tomokoko
"""

import pandas as pd
import matplotlib.pyplot as plt
import random

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)


import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2, random_state = 0)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test, y_pred)

print(cv.vocabulary_)

'''sum1 = 0
sum2 = 0
for i in range(0,1000):
    x = corpus[i].split()
    for j in range(0, len(x)):
        if x[j] == 'like':
            sum1 +=dataset['Liked'][i]
            sum2 += 1'''
            
