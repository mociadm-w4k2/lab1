#!/usr/bin/env python
import numpy as np
import pandas as pd

array = np.array([[1, 2, 3], [4, 5, 6]])

index = ['first row', 'last row']
columns = ['was', 'is', 'will be']

dataFrame = pd.DataFrame(array, index=index, columns=columns)

print dataFrame
