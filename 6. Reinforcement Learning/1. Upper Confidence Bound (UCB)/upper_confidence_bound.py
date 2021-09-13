# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 17:48:59 2021

@author: renzo
"""

# Upper Confidence Bound (UCB)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

# Algoritmo de UCB
import math
N = 10000   # Filas
d = 10      # Anuncios
number_of_selections = [0]*d
sums_of_rewards = [0]*d
ads_selected = [0]
total_reward = 0
for n in range(0, N):
    max_upper_bound = 0
    ad = 0
    for i in range(0, d):
        if number_of_selections[i] > 0:
            average_reward = sums_of_rewards[i]/number_of_selections[i]
            delta_i = math.sqrt(3/2*math.log(n+1)/number_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400     # Núm alto
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    number_of_selections[ad] = number_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward

# Histograma de resultados
plt.hist(ads_selected)
plt.title("Histograma de Anuncios")
plt.xlabel("ID del Anuncio")
plt.ylabel("Frecuencia de visualización")
plt.show()
