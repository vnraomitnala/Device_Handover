# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 18:35:46 2022

@author: Vijaya
"""
import librosa, librosa.display
import matplotlib.pyplot as plt
from dtw import dtw
import scipy
import math
import numpy as np
import pandas as pd
import os
import pickle
from scipy import signal
import scipy.io.wavfile
from collections import defaultdict

SAMPLE_RATE = 48000
blockLength =  2**14

dataset = {"microphone": [], "speaker": []  }
dataset2 = {"microphone": [], "speaker": []  }
dataset3 = {"microphone": [], "speaker": []  }

directory = os.fsencode('C:\\Users\\Vijaya\\PhD\\MicrophoneArray_Dataset\\tabletop11')

str1 = "C:/Users/Vijaya/PhD/MicrophoneArray_Dataset/tabletop11/tabletop11_speech05.wav"
str2 = "C:/Users/Vijaya/PhD/MicrophoneArray_Dataset/tabletop11/tabletop11_speech08.wav"

def shuffle(list, to_delete1, to_shuffle1, to_delete2, to_shuffle2):
   if len(list) > 0:
      list.remove(to_delete1) 
      list.remove(to_delete2)
      list.append(to_shuffle1)
      list.append(to_shuffle2)
   return

def shuffle2(list, to_delete1, to_shuffle1):
   if len(list) > 1:
      list.remove(to_delete1) 
      list.append(to_shuffle1)
   return

def MSCMeanForAllFrames(indata, blockLength):
    finished = False
    sample_index = 0
    block_length =0
    CxyMean = []

    while not finished:
        dataBlockL = indata[block_length+sample_index :,0]
        dataBlockR = indata[block_length+sample_index :,5]
        f, Cxy = signal.coherence(dataBlockR, dataBlockL, 48000, nperseg=1024)
        Cxy_mean = np.mean(Cxy)
        CxyMean.append(Cxy_mean)
        sample_index = sample_index + blockLength
        block_length = block_length + blockLength
        if sample_index >= int(len(data)):
            finished = True
    cleanedList = [x for x in CxyMean if x == x]
 
    return cleanedList        

labelsGlobal = []
for root, dirs, files in os.walk(directory):
          mscPrev = 0.0
          mscPrev1 = 0.0
          mscPrev2 = 0.0
          mscPrev3 = 0.0
          labels1 = []
          labels2 = []
          labels3 = []

          for file in files:
              str = os.path.join(root, file).decode("utf-8")
              print(str)
              if (str.find('speech') != -1  and str.find('tabletop') != -1 ):
                  samplerate, data = scipy.io.wavfile.read(str)
                  mscList = MSCMeanForAllFrames(data, blockLength)
                  # todo : do the list sorting. get the first and second best np.argsort
                  print("        ")
                  print("        ")
                  mscsort =  np.sort(np.array(mscList))[::-1]
                  mscCurrent1 = mscsort[0]
                  mscCurrent2 = mscsort[1]
                  mscCurrent3 = mscsort[2]
                  
                  print(mscCurrent1)
                  print(mscCurrent2)
                  print(mscCurrent3)
                  
                  if len(labels1) == 0:   
                      labels1.append(str[len(str)-15 : len(str)-13])
                  if len(labels2) == 0:           
                      labels2.append(str[len(str)-15 : len(str)-13])
                  if len(labels3) == 0:              
                      labels3.append(str[len(str)-15 : len(str)-13])                      

                    
                  mscMeanCurrent = np.mean(mscList)
                  if mscCurrent1 > mscPrev1:
                     print("Inside 1") 
                     mscPrev1 = mscCurrent1
                     #dataset["msc"].append(np.array(mscList).flatten())
                     if len(labels1) > 1:
                         shuffle2(labels1, labels1[1],  (str[len(str)-6 : len(str)-4]))
                         #labels[0]= mscList
                     else:
                          labels1.append(str[len(str)-6 : len(str)-4])
               
                          #labels[0] = mscList
                  elif mscCurrent2 > mscPrev2:
                     print("Inside 2") 
                     mscPrev2 = mscCurrent2
                     #dataset["msc"].append(np.array(mscList).flatten())
                     if len(labels2) > 1:
                         shuffle2(labels2, labels2[1],  (str[len(str)-6 : len(str)-4]))
                         #labels[0]= mscList
                     else:
                          labels2.append(str[len(str)-6 : len(str)-4])
                          #labels[0] = mscList   
                  elif mscCurrent3 > mscPrev3:
                     print("Inside 3")
                     mscPrev3 = mscCurrent3
                     #dataset["msc"].append(np.array(mscList).flatten())
                     if len(labels3) > 1:
                         shuffle2(labels3, labels3[1],  (str[len(str)-6 : len(str)-4]))
                         #labels[0]= mscList
                     else:
                          labels3.append(str[len(str)-6 : len(str)-4])
                          #labels[0] = mscList                           
                              

          print(labels1)
          print(labels2)
          print(labels3)
          
          dataset["microphone"].append(labels1[0])
          dataset2["microphone"].append(labels2[0])
          dataset3["microphone"].append(labels3[0])
          
          if len(labels1) == 2:
             dataset["speaker"].append(labels1[1])
          else: 
             dataset["speaker"].append("None")
             
          if len(labels2) == 2:   
             dataset2["speaker"].append(labels2[1])
          else: 
             dataset2["speaker"].append("None")   
             
          if len(labels3) == 2:   
             dataset3["speaker"].append(labels3[1])
          else: 
             dataset3["speaker"].append("None")              

print(labelsGlobal)
print(dataset)  
print(dataset2)
print(dataset3)

with open ('C:/Users/Vijaya/PhD/Audio_DataSetmsc_tmp2.pickle', 'wb' ) as f:
    pickle.dump(dataset, f)  

# =============================================================================
# df = pd.DataFrame(dataset.items(), columns=['microphone', 'speaker'])  
# 
# print(df)
# =============================================================================

