#!/usr/bin/env python
import pandas as pd
import numpy as np
import sklearn.preprocessing as sklp

names = ['class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash ', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

data = pd.read_csv('wine.csv', names=names)
array = data.values

X = array[:,1:]
Y = array[:,0]

scaler = sklp.StandardScaler().fit(X)
rescaledX = scaler.transform(X)

np.set_printoptions(precision=3)
print(X[:5,:5])
print(rescaledX[:5,:5])
