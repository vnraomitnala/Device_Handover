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

SAMPLE_RATE = 48000
blockLength =  2**14

dataset = {"microphone": [], "speaker": []  }

directory = os.fsencode('C:\\Users\\Vijaya\\PhD\\MicrophoneArray_Dataset\\')

def shuffle(list, to_delete1, to_shuffle1, to_delete2, to_shuffle2):
   if len(list) > 0:
      list.remove(to_delete1) 
      list.remove(to_delete2)
      list.append(to_shuffle1)
      list.append(to_shuffle2)
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
    return np.mean(cleanedList)        


for root, dirs, files in os.walk(directory):
          mscPrev = 0.0
          labels = []
          for file in files:
              str = os.path.join(root, file).decode("utf-8")
              if (str.find('speech') != -1 ):
                  print(str)
                  samplerate, data = scipy.io.wavfile.read(str)
                  mscMeanCurrent = MSCMeanForAllFrames(data, blockLength)
                  if mscMeanCurrent > mscPrev:
                     mscPrev = mscMeanCurrent
                     if len(labels) > 0:
                         length = str.find("speech")                               
                         microphoneValue = str[length-3 : length-1]
                         print(microphoneValue)
                         shuffle(labels, labels[0], microphoneValue, labels[1],  (str[len(str)-6 : len(str)-4]))
                     else:
                          length = str.find("speech")
                          microphoneValue = str[length-3 : length-1]
                          labels.append(microphoneValue)
                          labels.append(str[len(str)-6 : len(str)-4])

          if len(labels) > 0:
             dataset["microphone"].append(labels[0])
             dataset["speaker"].append(labels[1])

print(dataset)  

with open ('C:/Users/Vijaya/PhD/Audio_DataSetmsc2.pickle', 'wb' ) as f:
    pickle.dump(dataset, f)  

