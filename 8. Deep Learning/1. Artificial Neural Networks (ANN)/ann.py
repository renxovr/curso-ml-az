# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 01:38:19 2021

@author: renzo
"""

# Redes Neuronales Artificiales

# 1) Preprocesado

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Churn_Modelling.csv")

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [1])],   
    remainder='passthrough'                        
)
X = onehotencoder.fit_transform(X)
X = X[:, 1:]

# Dividir en entrenamiento y testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 2) Construir la RNA

import keras
from keras.models import Sequential
from keras.layers import Dense

# Iniciar RNA
classifier = Sequential()

# Añadir las capas de entrada y primera capa oculta
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))  # units -> experimentar, puede ser (11+1)/2

# Añadir la segunda capa oculta
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

# Añadir la capa de salida
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))  # Resultado binario - Sí/No

# Compilar la RNA
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Ajustar la RNA al Conjunto de Entrenamiento
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# 3) Evaluar modelo y calcular predicciones finales

# Predicción de los resultados con el conj de testing
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)   # Usar un umbral

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
