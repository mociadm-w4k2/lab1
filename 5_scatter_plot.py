#!/usr/bin/env python
import matplotlib.pyplot as plt
import pandas as pd

from pandas.tools.plotting import scatter_matrix

names = ['class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash ', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

data = pd.read_csv('wine.csv', names=names)
reducedData = data[['class', 'Color intensity', 'Alcohol', 'Hue']]

scatter_matrix(reducedData)

plt.show()
