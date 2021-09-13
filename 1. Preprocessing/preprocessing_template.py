# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 07:21:23 2021

@author: renzo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Data.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#Dividir en entrenamiento y testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Escalado de variables
'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
#El profe no pasa rangos, de frente todo el array *probar cualquiera*
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
'''











