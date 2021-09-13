# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 11:04:14 2021

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

#Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1,1))

#Ajustar la regresión con el dataset
from sklearn.svm import SVR
regression = SVR(kernel = "rbf")
regression.fit(X, y)


#Predicción del modelo con SVR
y_pred = regression.predict(sc_X.transform([[6.5], [7.5]]))
y_pred_inv = sc_y.inverse_transform(y_pred)

#Visualización de resultados con SVR
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color="red")
plt.plot(X_grid, regression.predict(X_grid), color="blue")
plt.title("Modelo de Regresión SVR")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo ($)")
plt.show()

#Visualización de resultados con SVR en valores originales
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color="red")
plt.plot(sc_X.inverse_transform(X_grid), sc_y.inverse_transform(regression.predict(X_grid)), color="blue")
plt.title("Modelo de Regresión SVR")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo ($)")
plt.show()
