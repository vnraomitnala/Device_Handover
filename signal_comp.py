# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 18:18:58 2021

@author: Vijaya
"""
import librosa, librosa.display
import matplotlib.pyplot as plt
from dtw import dtw
import scipy

#from scipy.spacial import Delaunay

soundpath = "C:\\Users\\Vijaya\\PhD\\MicrophoneArray_Dataset\\tabletop11\\"

#Loading audio files
y1, sr1 = librosa.load(soundpath+'tabletop11_speech06.wav') 
y2, sr2 = librosa.load(soundpath+'tabletop11_speech10.wav') 

blockLength =  1000

print(len(y1))

def getDataFrames(blockLength, indata):
    finished = False
    sample_index = 0
    data_blocks = []
    while not finished:
        data_block = indata[sample_index:blockLength+sample_index]
        data_blocks.append(data_block)
        sample_index = sample_index + blockLength
        if sample_index >= len(indata):
            finished = True
    return data_blocks

y1_frames = getDataFrames(blockLength, y1)
y2_frames = getDataFrames(blockLength, y2)

print(y1_frames[0])
print(y1_frames[1])

#Showing multiple plots using subplot
plt.subplot(1, 2, 1) 
mfcc1 = librosa.feature.mfcc(y1_frames[0],sr1, n_mfcc=13, n_fft=2048, hop_length=512)   #Computing MFCC values
librosa.display.specshow(mfcc1)

plt.subplot(1, 2, 2)
mfcc2 = librosa.feature.mfcc(y2_frames[0], sr2, n_mfcc=13, n_fft=2048, hop_length=512)
librosa.display.specshow(mfcc2)

l2_norm = lambda y1, y2: (y1 -y2) ** 2

dist= dtw(mfcc1.T, mfcc2.T, dist = l2_norm)
print("The normalized distance between the two : ",dist)   # 0 for similar audios 

# plt.imshow(cost.T, origin='lower', cmap=plt.get_cmap('gray'), interpolation='nearest')
# plt.plot(path[0], path[1], 'w')   #creating plot for DTW

plt.show()  #To display the plots graphically