#!/usr/bin/env python
import pandas as pd

from sklearn import neighbors
from sklearn import svm
from sklearn import neural_network
from sklearn import tree
from sklearn import naive_bayes

from sklearn.model_selection import cross_val_score

names = ['class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash ', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

data = pd.read_csv('wine.csv', names=names)
array = data.values

X = array[:,1:]
Y = array[:,0]

clfs = [
    neighbors.KNeighborsClassifier(),
    svm.SVC(kernel='linear'),
    tree.DecisionTreeClassifier(),
    naive_bayes.GaussianNB()
]

for clf in clfs:
    clf.fit(X, Y)
    scores = cross_val_score(clf, X, Y, cv=10)
    print 'Acc: %.3f' % scores.mean()
