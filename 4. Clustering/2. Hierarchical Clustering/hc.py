# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 17:59:13 2021

@author: renzo
"""

# Clustering Jerárquico

# Importar librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Agregar datos
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:, [3,4]].values

# Usar dendograma para encontrar n° óptimo de clústers
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method="ward"))
plt.title("Dendrograma")
plt.xlabel("Clientes")
plt.ylabel("Distancia Euclídea")
plt.show()

# Ajustar el clústering jerárquico
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity="euclidean", linkage="ward")
y_hc = hc.fit_predict(X)

# Visualización de los clústers
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=100, color="red", label="Clúster 1")
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=100, color="blue", label="Clúster 2")
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s=100, color="green", label="Clúster 3")
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s=100, color="cyan", label="Clúster 4")
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s=100, color="magenta", label="Clúster 5")
plt.title("Clúster de Clientes")
plt.xlabel("Ingresos Anuales (miles $)")
plt.ylabel("Puntuación de Gastos (1 - 100)")
plt.legend()
plt.show()
