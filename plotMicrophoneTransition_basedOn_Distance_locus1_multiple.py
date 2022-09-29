# -*- coding: utf-8 -*-
"""
Created on Sat May 28 12:35:50 2022

@author: Vijaya
"""

import math
import matplotlib.pyplot as plt
import pandas as pd
import pickle


with open ('locus_pos_list.pickle', 'rb') as f:
    locus_pos = pickle.load(f) 
    
standardRoom_mic_locs2 = [
    [1.5,3.5, 0.9], [5.5,3.5,0.9],  # mic 1  # mic 2
]

datasetDistance = []

for j in range(len(locus_pos)):
    datasetDistance2 = {"mic": [], "mic1-distance": [], "mic2-distance": []}
    for i in range(len(locus_pos[j])):
        mic1_distance = 0.0
        mic2_distance = 0.0
        for j in range(len(standardRoom_mic_locs2)):
            if j == 0:
               mic1_distance = math.dist(locus_pos[j][i] , standardRoom_mic_locs2[j])
               print(round(mic1_distance,2))
               datasetDistance2["mic1-distance"].append(round(mic1_distance,2))
            else:
                mic2_distance = math.dist(locus_pos[j][i], standardRoom_mic_locs2[j])
                print(round(mic2_distance,2))
                datasetDistance2["mic2-distance"].append(round(mic2_distance,2))
                
        if round(mic1_distance,2) < round(mic2_distance,2):
          datasetDistance2["mic"].append('01')
        else:
          datasetDistance2["mic"].append('02') 
    datasetDistance.append(datasetDistance2)     
print(datasetDistance)

with open ('C:/Users/Vijaya/PhD/locus_move_distance.pickle', 'wb' ) as f:
    pickle.dump(datasetDistance, f) 
