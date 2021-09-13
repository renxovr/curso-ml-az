# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 22:09:06 2021

@author: renzo
"""

# K-Means

# Importar librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Agregar datos
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:, [3,4]].values

# Método del codo para deternimar n° óptimo de clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title("Método del Codo")
plt.xlabel("Método de Clústers")
plt.ylabel("WCSS(k)")
plt.show()

# Aplicar el método k-means
kmeans = KMeans(n_clusters=5, init="k-means++", max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)

# Visualización de los clústers
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, color="red", label="Clúster 1")
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, color="blue", label="Clúster 2")
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, color="green", label="Clúster 3")
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, color="cyan", label="Clúster 4")
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, color="magenta", label="Clúster 5")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, color="yellow", label="Baricentros")
plt.title("Clúster de Clientes")
plt.xlabel("Ingresos Anuales (miles $)")
plt.ylabel("Puntuación de Gastos (1 - 100)")
plt.legend()
plt.show()
