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
from pandas import read_csv
from numpy import mean
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.api import SimpleExpSmoothing


SAMPLE_RATE = 16000
blockLength =  2**14
freqs = np.arange(0, 1 + 2**14 / 2) * 16000 / 2**14

dataset = {"speaker": [], "microphone": [], "coherence1": [] ,"coherence2": [],"coherence1_smooth": [], "coherence2_smooth": [] }

directory = 'C://Users//Vijaya//Documents//MobaXterm//home//tmp//3d//'

def double_exponential_smoothing(series, alpha, beta, n_preds=2):
    """
    Given a series, alpha, beta and n_preds (number of
    forecast/prediction steps), perform the prediction.
    """
    n_record = len(series)
    results = np.zeros(n_record + n_preds)

    # first value remains the same as series,
    # as there is no history to learn from;
    # and the initial trend is the slope/difference
    # between the first two value of the series
    level = series[0]
    results[0] = series[0]
   
    if n_record == 1:
        return series[0]
    
    trend = series[1] - series[0]
    for t in range(1, n_record + 1):
        if t >= n_record:
            # forecasting new points
            value = results[t - 1]
        else:
            value = series[t]

        previous_level = level
        level = alpha * value + (1 - alpha) * (level + trend)
        trend = beta * (level - previous_level) + (1 - beta) * trend 
        results[t] = level + trend

    # for forecasting beyond the first new point,
    # the level and trend is all fixed
    if n_preds > 1:
        results[n_record + 1:] = level + np.arange(2, n_preds + 1) * trend

    return round(results[-1],2)


def exponential_smoothing(series, alpha):
    """given a series and alpha, return series of expoentially smoothed points"""
    results = np.zeros_like(series)

    # first value remains the same as series,
    # as there is no history to learn from
    results[0] = series[0] 
    for t in range(1, len(series)):
        results[t] = alpha * series[t] + (1 - alpha) * results[t - 1]

    return round(results[-1],2)

def SimpleExpSmoothing(data):
    ses = SimpleExpSmoothing(data)
    alpha = 0.2
    model = ses.fit(smoothing_level = alpha, optimized = False)
    forcast = model.forecast(data.size())
    return forcast

def Smoothening(X):
    window = 3
    history = [X[i] for i in range(window)]
    test = [X[i] for i in range(window, len(X))]
    
    predictions = list()
    # walk forward over time steps in test
    for t in range(len(test)):
    	length = len(history)
    	yhat = mean([history[i] for i in range(length-window,length)])
    	obs = test[t]
    	predictions.append(yhat)
    	history.append(obs)
    return predictions   

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

def swap(list, to_delete1, to_shuffle1, to_delete2, to_shuffle2):
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
        dataBlockR = indata[block_length+sample_index :,1]
        
        y = lfilter(b, a, dataBlockL)
        mean_blockL = np.mean(10**(a/10))
        
        y = lfilter(b, a, dataBlockR)
        mean_blockR = np.mean(10**(a/10))
        
        if (mean_blockL) > 0.001 and (mean_blockR) > 0.001:
            if dataBlockL.size > 0 and dataBlockR.size > 0:
                kernel_size = 10
                kernel = np.ones(kernel_size) / kernel_size
                dataBlockL = np.convolve(dataBlockL, kernel, mode='same')
                dataBlockR = np.convolve(dataBlockR, kernel, mode='same')
               #dataBlockL =  savgol_filter(dataBlockL, 5, 2, mode='nearest')
                #dataBlockR =  savgol_filter(dataBlockR, 5, 2, mode='nearest')
                f, Cxy = signal.coherence(dataBlockR, dataBlockL, 16000, nperseg=1024)
# =============================================================================
#                 if len(Cxy) > 0:
#                    print(Cxy)
#                    Cxy = double_exponential_smoothing(Cxy.flatten(), 0.9, 0.1)
# =============================================================================
                cleanedList2 = [x for x in Cxy if x == x]   
                if len(cleanedList2) > 0:
                    Cxy_mean = np.mean(cleanedList2)
                CxyMean.append(Cxy_mean)
        sample_index = sample_index + blockLength
        block_length = block_length + blockLength
        if sample_index >= int(len(data)):
            finished = True
        cleanedList = [x for x in CxyMean if x == x]   

    return round(np.mean(cleanedList),2)     


def isStringMatched (index, fileStr, indata):
    found = False
   
    for i in range(0, len(indata),2):

        if (indata[i+1] == (fileStr[len(fileStr)-6 : len(fileStr)-4])):
        
            found = True
    return found

tmp = []

coherence_list = []
mic_list = []
coherence1 = []
coherence2 = []
cohenrence_smooth1 = []
cohenrence_smooth2 = []

for root, dirs, files in os.walk(directory):
          dirs.sort(key=int)

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

              coherence1_smooth = 0.0
              coherence2_smooth = 0.0
              
              for file in (files):
                  str = os.path.join(root, file)
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
                             coherence1_smooth = double_exponential_smoothing(coherence1, 0.1, 0.1)
                             print("coherence1" , coherence1)
                             
                             cohenrence_smooth1.append(coherence1_smooth)
                             print("coherence1_smooth" , cohenrence_smooth1)
                          else:
                             coherence2.append(mscMeanCurrent)
                             print("coherence2" , coherence2)
                             coherence2_smooth = double_exponential_smoothing(coherence2, 0.1,0.1)
                             cohenrence_smooth2.append(coherence2_smooth)
                             print("coherence2_smooth" , cohenrence_smooth2)
                          if mscMeanCurrent > mscPrev:
                             mscPrev = mscMeanCurrent                                                         

                             if len(labels) > 0:
                                 length = str.find("speech")                               
                                 microphoneValue = str[length-3 : length-1]
                                 swap(labels, labels[0], microphoneValue, labels[1],  (str[len(str)-6 : len(str)-4]))
                             else:
                                  length = str.find("speech")
                                  microphoneValue = str[length-3 : length-1]
                                  labels.append(microphoneValue)
                                  labels.append(str[len(str)-6 : len(str)-4])
                                 
                  file_index = file_index +1  

              print(labels)               

              index = index + 1  

              if len(labels) > 0:
                 if coherence1_smooth > coherence2_smooth:
                     labels[0] = '01'
                 else:
                     labels[0] = '02' 
                 dataset["microphone"].append(labels[0])
                 dataset["speaker"].append(labels[1])
                 labelsGlobal.append(labels[0])
                 labelsGlobal.append(labels[1])
dataset["coherence1"] = (coherence1)
dataset["coherence2"] = (coherence2)
dataset["coherence1_smooth"] = (cohenrence_smooth1)
dataset["coherence2_smooth"] = (cohenrence_smooth2)                
     
print(dataset)

with open ('C:/Users/Vijaya/PhD/Audio_DataSetmsc_3d_with_double_exp_smooth_0.8.pickle', 'wb' ) as f:
    pickle.dump(dataset, f)  



