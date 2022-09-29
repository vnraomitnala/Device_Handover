import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from scipy.io import wavfile
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
from shutil import rmtree

# The desired reverberation time and dimensions of the room

str1 = "examples_samples_guitar_16k_latest.wav"


blockLength =  2**14


rt60_tgt = 0.8 # seconds

room_dim = [10, 10, 2.4]  # meters

standard_room_dim = [7,5.5,2.4]  # meters

# import a mono wavfile as the source signal
# the sampling frequency should match that of the room
fs, audio = wavfile.read(str1)

# We invert Sabine's formula to obtain the parameters for the ISM simulator
e_absorption, max_order = pra.inverse_sabine(rt60_tgt, standard_room_dim)


standardRoom_mic_locs2 = [
    [1.0,3.5, 0.9], [5.5,3.5,0.9],  # mic 1  # mic 2
]


standardRoom_source_locs2 = [ [ 0.5 +i*0.05,1, 1.4] for i in range(111) ]


travel_time = [0+i*0.275 for i in range(111)]

length = len(audio)/fs

locus = standardRoom_source_locs2

rmtree('tmp/3d' , ignore_errors=True)

 
final_mic1 = []
final_mic2 = []
counter = 0
previous_signal = []

for j in range (len(locus)-1):

    for i in range(len(standardRoom_mic_locs2)):
        room = pra.ShoeBox(
            standard_room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order)
        
        room.add_microphone_array(pra.circular_microphone_array_xyplane(standardRoom_mic_locs2[i] , 2, 0, 0.1, room.fs))

                
        if travel_time[j+1] * fs >= len(audio):
            maxSample = len(audio)
        else:
            maxSample = np.floor(travel_time[j+1] * fs).astype(np.int)
            
        minSample = np.floor(travel_time[j] *fs).astype(np.int)
        
        overlap = np.floor(fs/100).astype(np.int)
               
        signal = audio[minSample  : maxSample]
        
        previous_signal = signal
        print(len(signal))
        print(len(previous_signal))
        
        if j >0:
            overlapped_signal = previous_signal[-int(len(signal)/2):]            
            signal = np.append(signal, overlapped_signal)
        
        print(len(signal))
        #signal = librosa.util.fix_length(signal,size=(len(signal) + np.max((len(room.rir[0][0]), len(room.rir[1][0])))))
        
        signal = np.hanning(len(signal))*signal
        signal = librosa.util.fix_length(signal,size=(len(signal) + 30710))
        
        room.add_source(locus[j], signal=signal)       
       
        room.simulate()
        
        #room.compute_rir()        
               
        if i == 0:
           
            tmp_final = final_mic1[minSample: len(final_mic1)]
            
            if len(tmp_final) == 0:
                tmp_final = np.array([[0],[0]])
            
            tmp_final = librosa.util.fix_length(tmp_final,size=(len(signal) + 30710))
            
            output = room.mic_array.signals
            
            output = librosa.util.fix_length(output,size=(len(signal) + 30710))
            
            tmp_final = tmp_final + output
            
            if minSample == 0:
                final_mic1 = output
            else:
                final_mic1 = librosa.util.fix_length(final_mic1,size=(minSample))
            
                final_mic1 = np.append(final_mic1,tmp_final, axis=1)
        else:
            tmp_final = final_mic2[minSample: len(final_mic2)]     
            
            if len(tmp_final) == 0:
                tmp_final = np.array([[0],[0]])
            
            tmp_final = librosa.util.fix_length(tmp_final,size=(len(signal) + 30710))
            
            output = room.mic_array.signals
            
            output = librosa.util.fix_length(output,size=(len(signal) + 30710))
            
            tmp_final = tmp_final + output
            
            if minSample == 0:
                final_mic2 = output
            else:
                final_mic2 = librosa.util.fix_length(final_mic2,size=(minSample))
            
                final_mic2 = np.append(final_mic2,tmp_final, axis=1)
   
os.makedirs("tmp/3d/", exist_ok = True)            
  
wavfile.write("tmp/3d/tabletop01_speech01.wav", room.fs, pra.normalize(final_mic1.T))
wavfile.write("tmp/3d/tabletop02_speech01.wav", room.fs, pra.normalize(final_mic2.T))

