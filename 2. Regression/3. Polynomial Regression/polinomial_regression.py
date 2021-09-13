# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 01:59:10 2021

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
#Ajustar la RL con el dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#Ajustar la RP con el dataset
from sklearn.preprocessing import PolynomialFeatures
pol_reg = PolynomialFeatures(degree = 4)
X_pol = pol_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_pol, y)

#Visualización RL
plt.scatter(X, y, color="red")
plt.plot(X, lin_reg.predict(X), color="blue")
plt.title("Modelo de RL")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo ($)")
plt.show()

#Visualización RP
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color="red")
#plt.plot(X, lin_reg_2.predict(pol_reg.fit_transform(X)), color="blue")
plt.plot(X_grid, lin_reg_2.predict(pol_reg.fit_transform(X_grid)), color="blue")
plt.title("Modelo de RP")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo ($)")
plt.show()

#Predicción del modelo
lin_reg.predict([[6.5], [7.5]])
lin_reg_2.predict(pol_reg.fit_transform([[6.5], [7.5]]))


