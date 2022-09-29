import librosa
import numpy as np
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


# import a mono wavfile as the source signal
# the sampling frequency should match that of the room
fs, audio = wavfile.read(str1)

print("audio size:", len(audio))

audio_len_sec = len(audio)/fs

print("audio_len_sec: ", audio_len_sec)

frame_len, hop_len = 475, 100

frames = librosa.util.frame(audio, frame_length=frame_len, hop_length=hop_len, axis=0)
windowed_frames = np.hanning(frame_len)*frames

print(len(frames))
print(len(windowed_frames))

   
final = windowed_frames[0]
counter = 0

for i in range(len(windowed_frames)-1):
    overlap = final[(hop_len+counter):(frame_len+counter)] + windowed_frames[i+1][0:(frame_len - hop_len)]
    print(overlap)
    final = librosa.util.fix_length(final,size=(counter+hop_len))
    final = np.concatenate((final,overlap))
    final = np.concatenate((final,windowed_frames[i+1][(frame_len - hop_len):frame_len]))
    counter += hop_len    

#wavfile.write( "tmp.wav", fs, pra.normalize(final))
wavfile.write( "tmp.wav", fs,final.astype(np.int16))

fs, audio2 = wavfile.read("tmp.wav")

audio_len2 = len(audio2)/fs
print(audio_len2)


# =============================================================================
# # Print frames
# for i, frame in enumerate(frames):
#     print("Frame {}: {}".format(i, frame))
# 
# =============================================================================
# =============================================================================
# # Print windowed frames
# for i, frame in enumerate(windowed_frames):
#     print("Win Frame {}: {}".format(i, np.round(frame, 3)))
# =============================================================================

