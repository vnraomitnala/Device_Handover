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


with open ('locus1_100.pickle', 'rb') as f:
    locus = pickle.load(f) 


blockLength =  2**14


rt60_tgt = 0.6 # seconds

room_dim = [10, 10, 2.4]  # meters

standard_room_dim = [7,5.5,2.4]  # meters
conf_room_dim = [10, 7, 2.7]

# import a mono wavfile as the source signal
# the sampling frequency should match that of the room
fs, audio = wavfile.read(str1)

audio_len = len(audio)/fs

print(audio_len)

# We invert Sabine's formula to obtain the parameters for the ISM simulator
e_absorption, max_order = pra.inverse_sabine(rt60_tgt, conf_room_dim)


standardRoom_mic_locs2 = [
    [1.0,3.5, 0.9], [5.5,3.5,0.9],  # mic 1  # mic 2
]

confRoom_mic_locs = [
    
     [2,4,0.9], [5,4,0.9],  # mic 1  # mic 2
     
 ]


standardRoom_source_locs2 = [ [ 0.5 +i*0.05,1, 1.4] for i in range(111) ]

standardRoom_source_locs_tmp = [ [ 0.5 +i*0.5,1, 1.4] for i in range(10) ]

time = []

confRoom_source_locs = [ [0.5 +i*0.05, 1, 1.4] for i in range(115) ]

locus = confRoom_source_locs

k = 0
rmtree('tmp/3d' , ignore_errors=True)

for j in range (len(locus)):
    for i in range(2):
        room = pra.ShoeBox(
            conf_room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order)
        
        room.add_source(locus[j], signal=audio, delay=0.5)
        room.add_microphone_array(pra.circular_microphone_array_xyplane(confRoom_mic_locs[i] , 2, 0, 0.1, room.fs))
        room.simulate()
# =============================================================================
#         room.plot(freq=[1000, 2000, 4000, 8000], img_order=0)
#         plt.show()
# =============================================================================
        
        str3 = "tmp/3d/" + str(k) + "/"
        os.makedirs(str3, exist_ok = True)
        
        if i == 0:
            str4 = str3 + "tabletop01_speech01.wav"
            wavfile.write( str4, room.fs, pra.normalize(room.mic_array.signals.T))
            plt.plot(room.mic_array.signals[0, :])
        else:
            str5 = str3 + "tabletop02_speech01.wav"
            wavfile.write(str5, room.fs, pra.normalize(room.mic_array.signals.T))
            plt.plot(room.mic_array.signals[0, :])
            

    k= k+1
