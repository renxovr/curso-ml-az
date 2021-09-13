# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 12:53:29 2021

@author: renzo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Position_Salaries.csv")

X = dataset.iloc[:, 1:2].values #Se necesita que sea una matriz
y = dataset.iloc[:, 2].values

#Dividir en entrenamiento y testing
'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
'''

#Ajustar la regresión con el dataset
from sklearn.tree import DecisionTreeRegressor
regression = DecisionTreeRegressor(random_state=0)
regression.fit(X, y)

#Predicción del modelo
y_pred = regression.predict([[6.5], [7.5]])

#Visualización
#X_grid = np.arange(min(X), max(X), 0.1)
#X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color="red")
plt.plot(X, regression.predict(X), color="blue")
plt.title("Modelo de Árbol de Decisión")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo ($)")
plt.show()
