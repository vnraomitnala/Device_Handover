# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 18:18:58 2021

@author: Vijaya
"""
import librosa, librosa.display
import matplotlib.pyplot as plt
from dtw import dtw
import scipy
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import r2_score
from sklearn import linear_model

DURATION = 61 # 61 sec
SAMPLE_RATE = 48000

# =============================================================================
# TOTAL_SAMPLES = 61 * 22050 
# 
# NUM_SEGMENTS = 5
# 
# num_samples_per_segment =  int(TOTAL_SAMPLES/NUM_SEGMENTS)
# 
# hop_length = 512
# 
# expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment/hop_length)
# =============================================================================

soundpath = "C:\\Users\\Vijaya\\PhD\\MicrophoneArray_Dataset\\tabletop11\\"

#Loading audio files
y1, sr1 = librosa.load(soundpath+'tabletop11_speech06.wav', sr= SAMPLE_RATE, mono=False) 
y2, sr2 = librosa.load(soundpath+'tabletop11_speech10.wav', sr= SAMPLE_RATE, mono=False) 

blockLength =  2**14

dataSet = {"labels": [], "mfcc": []  }

def getDataFrames(blockLength, indata):
    data_blocks = []
    for i in range(0,8):
        finished = False
        sample_index = 0
        data_block = []
        while not finished:
            datablock = []
            datablock = indata[i][sample_index:blockLength+sample_index]
            df = pd.DataFrame(datablock)
            data_block.append(datablock)
            sample_index = sample_index + blockLength
            if sample_index >= int(len(indata[0])):
                finished = True
        data_blocks.append(data_block)
        return data_blocks

def getDataBlocks(blockLength, indata):
    finished = False
    sample_index = 0
    data_blocks = []
    while not finished:
        data_block = indata[:, sample_index:blockLength+sample_index]
        data_blocks.append(data_block)
        sample_index = sample_index + blockLength
        if sample_index >= int(len(indata[0])):
            finished = True
    return data_blocks

y1_frames = getDataBlocks(blockLength, y1)
# df = pd.DataFrame(y1_frames)
# print(df)
#y2_frames = getDataFrames(blockLength, y2)

#Showing multiple plots using subplot
plt.subplot(1, 2, 1) 

#print(y1_frames[0]) 


# =============================================================================
# for item in range(0,8):
#     mfccs = []
#     for i in range(0,179):
#         mfcc = librosa.feature.mfcc(df[i][item],sr1, n_mfcc=13, n_fft=blockLength, hop_length=blockLength)   #Computing MFCC values
#         mfccs.append(mfcc.T)
#     dataSet["mfcc"].append(mfccs)
#     dataSet["labels"].append(6)
# =============================================================================
 
mfccs = []   
for i in range(0,179):
   mfcc = librosa.feature.mfcc(np.array(y1_frames[i]),SAMPLE_RATE, n_mfcc=13, n_fft=blockLength, hop_length=blockLength)   #Computing MFCC values
   #mfccs.append(mfcc.T)
   dataSet["mfcc"].append(mfcc.T)
   dataSet["labels"].append(6)

X = np.array(dataSet["mfcc"])
y = np.array(dataSet["labels"])


y.shape = (179,1)

# print(X.shape) # 83 rows, columns = 8*13
# print(y.shape) # 83 rows, column = 1 

 # create train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

modelRF = RandomForestClassifier(n_estimators=10)
modelRF.fit(X_train, y_train)

y_pred = modelRF.predict(X_test)

print("classifier accuracy:", accuracy_score(y_test, y_pred))

    # build network topology
model = keras.Sequential([

        # input layer
        keras.layers.Flatten(input_shape=(1346,1)),

        # 1st dense layer
        keras.layers.Dense(512, activation='relu'),

        # 2nd dense layer
        keras.layers.Dense(256, activation='relu'),

        # 3rd dense layer
        keras.layers.Dense(64, activation='relu'),

        # output layer
        keras.layers.Dense(1, activation='softmax')
    ])
    
  # compile model
optimiser = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

model.summary()

    # train model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=50)

    
# =============================================================================
# #librosa.display.specshow(mfcc1)
# 
# plt.subplot(1, 2, 2)
# mfcc2 = librosa.feature.mfcc(y2_frames[0], sr2, n_mfcc=13, n_fft=2048, hop_length=512)
# librosa.display.specshow(mfcc2)
# 
# l2_norm = lambda y1, y2: (y1 -y2) ** 2
# 
# # dist= dtw(mfcc1.T, mfcc2.T, dist = l2_norm)
# # print("The normalized distance between the two : ",dist)   # 0 for similar audios 
# 
# # plt.imshow(cost.T, origin='lower', cmap=plt.get_cmap('gray'), interpolation='nearest')
# # plt.plot(path[0], path[1], 'w')   #creating plot for DTW
# 
# plt.show()  #To display the plots graphically
# =============================================================================
