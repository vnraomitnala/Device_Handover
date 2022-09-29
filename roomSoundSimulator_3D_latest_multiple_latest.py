import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
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
from shutil import rmtree

# The desired reverberation time and dimensions of the room

str1 = "examples_samples_guitar_16k.wav"

with open ('locus_pos_list.pickle', 'rb') as f:
    locus_pos = pickle.load(f) 


blockLength =  2**14


rt60_tgt = 0.8 # seconds

room_dim = [10, 10, 2.4]  # meters

standard_room_dim = [7,5.5,2.4]  # meters
conf_room_dim = [10, 7, 2.7]

# import a mono wavfile as the source signal
# the sampling frequency should match that of the room
fs, audio = wavfile.read(str1)

# We invert Sabine's formula to obtain the parameters for the ISM simulator
e_absorption, max_order = pra.inverse_sabine(rt60_tgt, standard_room_dim)


standardRoom_mic_locs2 = [
    [1.0,3.5, 0.9], [5.5,3.5,0.9],  # mic 1  # mic 2
]

datasetFinal = []

for l in range(len(locus_pos)):
    print(l)
    locus = locus_pos[l]
    k = 0

    rmtree('tmp/3d' , ignore_errors=True)
    
    for j in range (len(locus)):
        
 

        for i in range(2):
            room = pra.ShoeBox(
                standard_room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order)
            
            room.add_source(locus[j], signal=audio, delay=0.5)
            room.add_microphone_array(pra.circular_microphone_array_xyplane(standardRoom_mic_locs2[i] , 2, 0, 0.1, room.fs))
            room.simulate()
            ss =  f'{k}'
            str3 = "tmp/3d/" + ss + "/"
            os.makedirs(str3, exist_ok = True)
            
            if i == 0:
                str4 = str3 + "tabletop01_speech01.wav"
                wavfile.write( str4, room.fs, pra.normalize(room.mic_array.signals.T))
#                plt.plot(room.mic_array.signals[0, :])
            else:
                str5 = str3 + "tabletop02_speech01.wav"
                wavfile.write(str5, room.fs, pra.normalize(room.mic_array.signals.T))
#                plt.plot(room.mic_array.signals[0, :])
                
    
        k= k+1
    
    SAMPLE_RATE = 16000
    blockLength =  2**14
    freqs = np.arange(0, 1 + 2**14 / 2) * 16000 / 2**14
    
    dataset = {"microphone_without_smooth": [], "microphone_with_smooth": [],
               "mic1_coherence_without_smooth": [] ,"mic2_coherence_without_smooth": [],
               "mic1_coherence_with_smooth": [], "mic2_coherence_with_smooth": [] }
    
    directory = 'tmp//3d//'
    
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
        
        trend = beta *(series[1] - series[0])
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
    
    def MicsSwap(list, to_delete1, to_shuffle1, to_delete2, to_shuffle2):
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
                    f, Cxy = signal.coherence(dataBlockR, dataBlockL, 16000, nperseg=1024)
                    
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
    
    mic1_coherence_without_smooth = []
    mic2_coherence_without_smooth = []
    mic1_coherence_with_smooth = []
    mic2_coherence_with_smooth = []
    
    for root, dirs, files in os.walk(directory):
              dirs.sort(key=int)
              file_index=0
    
              coherence1_without_smooth = 0.0
              coherence2_without_smooth = 0.0
              coherence1_with_smooth = 0.0
              coherence2_with_smooth = 0.0
              
              for file in (files):
                str = os.path.join(root, file)
                if (str.find('speech') != -1 ):
                       # print(str)
                        data, samplerate  = sf.read(str)  
                                               
    
                        if file_index == 0:
                           coherence1_without_smooth = MSCMeanForAllFrames(data, blockLength, samplerate) 
                           mic1_coherence_without_smooth.append(coherence1_without_smooth)
                           
                           
                           
                           coherence1_with_smooth = double_exponential_smoothing(mic1_coherence_without_smooth, 0.05, 0.01)                     
                           mic1_coherence_with_smooth.append(coherence1_with_smooth)
                           
                           
                        else:
                           
                           coherence2_without_smooth = MSCMeanForAllFrames(data, blockLength, samplerate)  
                           mic2_coherence_without_smooth.append(coherence2_without_smooth)
                           
                               
                           coherence2_with_smooth = double_exponential_smoothing(mic2_coherence_without_smooth, 0.05,0.01)
                           mic2_coherence_with_smooth.append(coherence2_with_smooth)
                           
                               
                file_index = file_index +1  
              
              if coherence1_without_smooth > 0.0:
                  if coherence1_without_smooth > coherence2_without_smooth:

                        dataset["microphone_without_smooth"].append('01')
                  else:
                        dataset["microphone_without_smooth"].append('02')
              
              if coherence1_with_smooth > 0.0:  
                  
                  if coherence1_with_smooth > coherence2_with_smooth:
                      dataset["microphone_with_smooth"].append('01')
                  else:
                      dataset["microphone_with_smooth"].append('02')
    
    dataset["mic1_coherence_without_smooth"] = mic1_coherence_without_smooth
    dataset["mic2_coherence_without_smooth"] = mic2_coherence_without_smooth
    dataset["mic1_coherence_with_smooth"] = mic1_coherence_with_smooth
    dataset["mic2_coherence_with_smooth"] = mic2_coherence_with_smooth                

    datasetFinal.append(dataset)         


with open ('datasetMSCFinal_list_latest.pickle', 'wb' ) as f:
    pickle.dump(datasetFinal, f) 
