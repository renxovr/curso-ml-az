# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 07:18:14 2021

@author: renzo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Data.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#Tratamiento de datos faltantes
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

