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

with open ('locus1_100.pickle', 'rb') as f:
    locus1 = pickle.load(f) 
    

print(locus1)

# =============================================================================
# locus1 = [[1.0,2.0,1.4],[1.0,3.0,1.4],[1.0,4.0,1.4],[2.0,4.0,1.4],
#           [3.0,4.0,1.4],[4.0,3.0,1.4],[3.0,4.0,1.4],[3.0,4.5,1.4]]
# =============================================================================


standardRoom_mic_locs2 = [
    [1.5,3.5, 0.9], [5.5,3.5,0.9],  # mic 1  # mic 2
]

standardRoom_source_locs2 = [ [ 0.5 +i*0.05,1, 1.4] for i in range(111) ]

locus1 = standardRoom_source_locs2


for i in range(len(locus1)):
    mic1_distance = 0.0
    mic2_distance = 0.0
    for j in range(len(standardRoom_mic_locs2)):
        if j == 0:
           mic1_distance = math.dist(locus1[i] , standardRoom_mic_locs2[j])
           print(round(mic1_distance,2))
           datasetDistance["mic1-distance"].append(round(mic1_distance,2))
        else:
            mic2_distance = math.dist(locus1[i], standardRoom_mic_locs2[j])
            print(round(mic2_distance,2))
            datasetDistance["mic2-distance"].append(round(mic2_distance,2))
            
    if round(mic1_distance,2) < round(mic2_distance,2):
      datasetDistance["mic"].append('01')
    else:
      datasetDistance["mic"].append('02') 

print(datasetDistance)

with open ('C:/Users/Vijaya/PhD/1.0_SR2_linear_move_dis.pickle', 'wb' ) as f:
    pickle.dump(datasetDistance, f) 


# Plotting ...loc
y= []

for i in range(len(locus1)):
    y.append(locus1[i][0])
    
xx = []
yy = []

for i in range(len(locus1)):
    xx.append(locus1[i][0])
    
for i in range(len(locus1)):
    yy.append(locus1[i][1])    
    
plt.subplot(2, 1, 1)
plt.plot(yy,xx,marker='x', markersize=20, color='r')

plt.subplot(2, 1, 2)
plt.plot(y, (datasetDistance['mic']), c ="blue", label='Ideal')

plt.xlabel('$x$ (distance)')
plt.ylabel('$y$ (transition)')

plt.grid()
plt.legend()

plt.show()