# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 07:17:51 2021

@author: renzo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Data.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#Codificar datos categóricos
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],   
    remainder='passthrough'                        
)
X = np.array(ct.fit_transform(X), dtype=np.float)

from sklearn import preprocessing
le_y = preprocessing.LabelEncoder()
y = le_y.fit_transform(y)

### Otro ejemplo en part 8. deep learning - ann