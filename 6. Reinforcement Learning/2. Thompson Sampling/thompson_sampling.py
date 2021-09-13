# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 21:18:36 2021

@author: renzo
"""

# Muestreo Thompson

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

# Algoritmo de Muestreo Thompson
import random
N = 10000   # Filas
d = 10      # Anuncios
number_of_rewards_1 = [0]*d
number_of_rewards_0 = [0]*d
ads_selected = [0]
total_reward = 0
for n in range(0, N):
    max_random = 0
    ad = 0
    for i in range(0, d):
        random_beta = random.betavariate(number_of_rewards_1[i]+1, number_of_rewards_0[i]+1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        number_of_rewards_1[ad] = number_of_rewards_1[ad]+1
    else:
        number_of_rewards_0[ad] = number_of_rewards_0[ad]+1
    total_reward = total_reward + reward

# Histograma de resultados
plt.hist(ads_selected)
plt.title("Histograma de Anuncios")
plt.xlabel("ID del Anuncio")
plt.ylabel("Frecuencia de visualización")
plt.show()






