# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 21:39:26 2021

@author: renzo
"""

# Apriori

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Market_Basket_Optimisation.csv", header=None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

# Entrenar el algoritmo Apriori
from apyori import apriori
# min_support: veces que aparece en el dataset sobre total, la data es de una semana 
# y se quiere que aparezca 3 veces por día = 3*7 / 3500 aprox 0.003
# jugar con min_confidence y min_lift
rules = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2)

# Visualización
results = list(rules)
results[0]
