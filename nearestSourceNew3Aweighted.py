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
import sys
from scipy.signal import lfilter
import numpy as np
import soundfile as sf

from numpy import pi, polymul
from scipy.signal import bilinear
from scipy.signal import savgol_filter
import skfda

SAMPLE_RATE = 48000
blockLength =  2**14
freqs = np.arange(0, 1 + 2**14 / 2) * 48000 / 2**14

dataset = {"speaker": [], "microphone": [], "coherence1": [], "micList1": []  
           ,"coherence2": [], "micList2": []}

directory = os.fsencode('C://Users//Vijaya//Documents//MobaXterm//home//tmp//3d')

def A_weighting(fs):
    """Design of an A-weighting filter.
    b, a = A_weighting(fs) designs a digital A-weighting filter for
    sampling frequency `fs`. Usage: y = scipy.signal.lfilter(b, a, x).
    Warning: `fs` should normally be higher than 20 kHz. For example,
    fs = 48000 yields a class 1-compliant filter.
    References:
       [1] IEC/CD 1672: Electroacoustics-Sound Level Meters, Nov. 1996.
    """
    # Definition of analog A-weighting filter according to IEC/CD 1672.
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    A1000 = 1.9997

    NUMs = [(2*pi * f4)**2 * (10**(A1000/20)), 0, 0, 0, 0]
    DENs = polymul([1, 4*pi * f4, (2*pi * f4)**2],
                   [1, 4*pi * f1, (2*pi * f1)**2])
    DENs = polymul(polymul(DENs, [1, 2*pi * f3]),
                                 [1, 2*pi * f2])

    # Use the bilinear transformation to get the digital filter.
    # (Octave, MATLAB, and PyLab disagree about Fs vs 1/Fs)
    return bilinear(NUMs, DENs, fs)

def rms_flat(a):  # from matplotlib.mlab
    """
    Return the root mean square of all the elements of *a*, flattened out.
    """
    return np.sqrt(np.mean(np.absolute(a)**2))

def shuffle(list, to_delete1, to_shuffle1, to_delete2, to_shuffle2):
   if len(list) > 0:
      list.remove(to_delete1) 
      list.remove(to_delete2)
      list.append(to_shuffle1)
      
      list.append(to_shuffle2)
   return

def MSCMeanForAllFrames(indata, blockLength, fs):
    finished = False
    sample_index = 0
    block_length =0
    CxyMean = []

    while not finished:
        b, a = A_weighting(fs)
        dataBlockL = indata[block_length+sample_index :,0]
        dataBlockR = indata[block_length+sample_index :,5]
        
        y = lfilter(b, a, dataBlockL)
        mean_blockL = np.mean(10**(a/10))
        
        y = lfilter(b, a, dataBlockR)
        mean_blockR = np.mean(10**(a/10))
        
        if (mean_blockL) > 0.001 and (mean_blockR) > 0.001:
            #dataBlockL =  savgol_filter(dataBlockL, 5, 2, mode='nearest')
            #dataBlockR =  savgol_filter(dataBlockR, 5, 2, mode='nearest')
            f, Cxy = signal.coherence(dataBlockR, dataBlockL, 16000, nperseg=1024)
            Cxy_mean = np.mean(Cxy)
            CxyMean.append(Cxy_mean)
            sample_index = sample_index + blockLength
            block_length = block_length + blockLength
            if sample_index >= int(len(data)):
                finished = True
        cleanedList = [x for x in CxyMean if x == x]        
    return round(np.mean(cleanedList),3)      


def isStringMatched (index, fileStr, indata):
    found = False
   
    for i in range(0, len(indata),2):

        if (indata[i+1] == (fileStr[len(fileStr)-6 : len(fileStr)-4])):
        
            found = True
    return found

tmp = []

coherence_list = []
mic_list = []

for root, dirs, files in os.walk(directory):

          if(len(files) > 3):
              toIter = 1
          else:
              toIter = 1 #len(files)
          index = 0

          labelsGlobal = []

          for i in range (0, toIter):
              labels = []
              mscPrev = 0.0
              file_index=0
              coherence1 = []
              coherence2 = []
              mic1 = []
              mic2 = []
              for file in files:
                  print(file_index)
                  str = os.path.join(root, file).decode("utf-8")
                  if (str.find('speech') != -1 ):
                      if index == 0:
                         datasetIndex = 0
                      else:
                         datasetIndex = index-1 
                      if len(labelsGlobal) > 0 and isStringMatched(index, str,labelsGlobal ):
                          print("breaking ", str )
                      else:
                          print(str)
                          data, samplerate  = sf.read(str)
                          mscMeanCurrent = MSCMeanForAllFrames(data, blockLength, samplerate)
                          print(mscMeanCurrent)
                          if file_index == 0:
                             coherence1.append(mscMeanCurrent)
                             mic1.append('01')
                          else:
                             coherence2.append(mscMeanCurrent)
                             mic2.append('02')
                      
                          if mscMeanCurrent > mscPrev:
                             mscPrev = mscMeanCurrent                                                         

                             if len(labels) > 0:
                                 length = str.find("speech")                               
                                 microphoneValue = str[length-3 : length-1]
                                 shuffle(labels, labels[0], microphoneValue, labels[1],  (str[len(str)-6 : len(str)-4]))
                             else:
                                  length = str.find("speech")
                                  microphoneValue = str[length-3 : length-1]
                                  labels.append(microphoneValue)
                                  labels.append(str[len(str)-6 : len(str)-4])
                  file_index = file_index +1  

              print(labels)               
                 
              index = index + 1  

              if len(labels) > 0:
                 dataset["microphone"].append(labels[0])
                 dataset["speaker"].append(labels[1])
                 dataset["coherence1"].append(coherence1[0])
                 dataset["coherence2"].append(coherence2[0])
                 dataset["micList1"].append(mic1[0])
                 dataset["micList2"].append(mic2[0])
                 labelsGlobal.append(labels[0])
                 labelsGlobal.append(labels[1])
          tmp.append(dataset)
     

print(dataset)

with open ('C:/Users/Vijaya/PhD/Audio_DataSetmsc_3d_1.0.pickle', 'wb' ) as f:
    pickle.dump(dataset, f)  



