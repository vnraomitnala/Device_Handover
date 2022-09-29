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

str1 = "examples_samples_guitar_16k.wav"


blockLength =  2**14


rt60_tgt = 0.6 # seconds


standard_room_dim = [7,5.5,2.4]  # meters


# import a mono wavfile as the source signal
# the sampling frequency should match that of the room
fs, audio = wavfile.read(str1)

audio_len = len(audio)/fs

print(audio_len)

# We invert Sabine's formula to obtain the parameters for the ISM simulator
e_absorption, max_order = pra.inverse_sabine(rt60_tgt, standard_room_dim)


standardRoom_mic_locs2 = [
    [1.0,3.5, 0.9], [5.5,3.5,0.9],  # mic 1  # mic 2
]

confRoom_mic_locs = [
    
     [2,4,0.9], [5,4,0.9],  # mic 1  # mic 2
     
 ]


standardRoom_source_locs2 = [ [ 0.5 +i*0.05,1, 1.4] for i in range(111) ]

locus = standardRoom_source_locs2

k = 0
rmtree('tmp/3d' , ignore_errors=True)

rir_mic1 = []
rir_mic2 = []

for j in range (len(locus)):
    for i in range(len(standardRoom_mic_locs2)):
        room = pra.ShoeBox(
            standard_room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order)
        
        room.add_source(locus[j], signal=audio)
        room.add_microphone_array(pra.circular_microphone_array_xyplane(standardRoom_mic_locs2[i] , 2, 0, 0.1, room.fs))
        
        room.compute_rir()
 
        if i == 0:
            max_rir = np.max((len(room.rir[0][0]), len(room.rir[1][0])))
            rir_mic1.append(max_rir)
        else:
            max_rir = np.max((len(room.rir[0][0]), len(room.rir[1][0])))
            rir_mic2.append(max_rir)

    k= k+1
    
print(max(rir_mic1))  
print(max(rir_mic2))    

