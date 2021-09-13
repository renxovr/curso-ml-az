# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 13:26:52 2021

@author: renzo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Salary_Data.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#Dividir en entrenamiento y testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#Crear modelo de RL simple con conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

#Predecir el conjunto de test
y_pred = regression.predict(X_test)

#Visualizar los datos de entrenamiento
plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, regression.predict(X_train), color="blue")
plt.title("Sueldo vs Años de exp. (Conjunto de entrenamiento)")
plt.xlabel("Años de exp.")
plt.ylabel("Sueldo ($)")
plt.show()

#Visualizar los datos de test
plt.scatter(X_test, y_test, color="red")
plt.plot(X_train, regression.predict(X_train), color="blue") #no hay diferencia si usamos la de test
plt.title("Sueldo vs Años de exp. (Conjunto de testing)")
plt.xlabel("Años de exp.")
plt.ylabel("Sueldo ($)")
plt.show()