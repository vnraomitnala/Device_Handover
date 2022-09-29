# -*- coding: utf-8 -*-
"""
Created on Sat May 28 12:35:50 2022

@author: Vijaya
"""

import math
import matplotlib.pyplot as plt
import pandas as pd
import pickle


datasetDistance = {"mic": [], "mic1-distance": [], "mic2-distance": []}

confRoom_source_locs = [ [0.5 +i*0.05, 1, 1.4] for i in range(115) ]
standardRoom_source_locs = [ [0.5 +i*0.05, 1, 1.4] for i in range(101) ]
standardRoom_source_locs2 = [ [0.5 +i*0.05, 1, 1.4] for i in range(111) ]

locus1 = [[1.0,2.0],[1.0,3.0],[1.0,4],[2.0,4.0],
          [3.0,4.0],[4.0,3.0],[3.0,4.0],[3.0,4.5]]

standardRoom_mic_locs = [
    [1.5,3.5,0.9], [3.5,3.5,0.9],  # mic 1  # mic 2
]

standardRoom_mic_locs2 = [
    [1.5,3.5, 0.9], [5.5,3.5,0.9],  # mic 1  # mic 2
]

confRoom_mic_locs = [
     [2,4,0.9], [5, 4,0.9],  # mic 1  # mic 2
 ]


for i in range(len(standardRoom_source_locs2)):
    mic1_distance = 0.0
    mic2_distance = 0.0
    for j in range(len(standardRoom_mic_locs2)):
        if j == 0:
           mic1_distance = math.dist(standardRoom_source_locs2[i] , standardRoom_mic_locs2[j])
           print(round(mic1_distance,2))
           datasetDistance["mic1-distance"].append(round(mic1_distance,2))
        else:
            mic2_distance = math.dist(standardRoom_source_locs2[i], standardRoom_mic_locs2[j])
            print(round(mic2_distance,2))
            datasetDistance["mic2-distance"].append(round(mic2_distance,2))
            
    if round(mic1_distance,2) < round(mic2_distance,2):
      datasetDistance["mic"].append('01')
    else:
      datasetDistance["mic"].append('02') 

print(datasetDistance)

with open ('C:/Users/Vijaya/PhD/Mic_Trancision_Based_On_Distance_SR2.pickle', 'wb' ) as f:
    pickle.dump(datasetDistance, f) 


# Plotting ...
y= []

for i in range(len(standardRoom_source_locs2)):
    y.append(standardRoom_source_locs2[i][0])

plt.subplot(3, 1, 1)
plt.plot(y, (datasetDistance['mic']), c ="blue", label='Ideal')

plt.xlabel('$x$ (distance)')
plt.ylabel('$y$ (transition)')

plt.grid()
plt.legend()


plt.show()