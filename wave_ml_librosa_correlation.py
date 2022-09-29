import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import scipy.io
import scipy.io.wavfile
from scipy.fft import fft
#from scipy.io import wavefile
import scipy.signal as signal
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import r2_score
from sklearn import linear_model
import csv
import pickle
import scipy.stats
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
from matplotlib.backends.backend_pdf import PdfPages
import librosa, librosa.display
import os

# =============================================================================
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import cross_val_score
# from sklearn.preprocessing import LabelEncoder
# from keras.optimizers import SGD
# 
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.layers import Dense, Dropout,Activation, Flatten
# =============================================================================
#from tensorflow.keras.optimizers import adam

librosa_list_new = []
directory = os.fsencode('C:\\Users\\Vijaya\\PhD\\MicrophoneArray_Dataset\\')
    
# =============================================================================
# for root, dirs, files in os.walk(directory):
#          for file in files:
#              print(os.path.join(root, file).decode("utf-8"))
#              str = os.path.join(root, file).decode("utf-8")
#              librosa_data, librosa_samplerate = librosa.load(str, sr = None, mono= False)
#              print(librosa_data)
#              librosa_list_new.append(librosa_data)
# 
# print(librosa_list_new)             
# =============================================================================

soundpath = "C:\\Users\\Vijaya\\PhD\\MicrophoneArray_Dataset\\tabletop11\\"

samplerate, data = scipy.io.wavfile.read('C:\\Users\\Vijaya\\PhD\\MicrophoneArray_Dataset\\tabletop11\\tabletop11_mix.wav')

librosa_data6, sample_rate6 = librosa.load(soundpath+"tabletop11_speech06.wav",sr=None,mono=False)
librosa_data10, sample_rate10 = librosa.load(soundpath+"tabletop11_speech10.wav",sr=None,mono=False)

print(librosa_data6.shape)

librosa_traing6 = librosa_data6[:, 0:int(len(librosa_data6[0])/2)-1]
librosa_testing6 = librosa_data6[:, int(len(librosa_data6[0])/2): int(len(librosa_data6[0])) ]

print(librosa_traing6.shape)
print(librosa_testing6.shape)

librosa_traing10 = librosa_data10[:, 0:int(len(librosa_data10[0])/2)-1]
librosa_testing10 = librosa_data10[:, int(len(librosa_data10[0])/2): int(len(librosa_data10[0])) ]

print(librosa_traing10.shape)
print(librosa_testing10.shape)


data_trianing_blocks6 = []
data_testing_blocks6 = []
data_trianing_blocks10 = []
data_testing_blocks10 = []

blockLength =  1000

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

data_trianing_blocks6 = getDataBlocks(blockLength, librosa_traing6)   
data_testing_blocks6 = getDataBlocks(blockLength, librosa_testing6)   

data_trianing_blocks10 = getDataBlocks(blockLength, librosa_traing10)   
data_testing_blocks10 = getDataBlocks(blockLength, librosa_testing10)                       

print("data_trianing_blocks6", data_trianing_blocks6[732].shape)
print("data_trianing_blocks10", len(data_trianing_blocks10))
#print("data_trianing_blocks[0]", data_trianing_blocks[0])
#print("data_testing_blocks", data_testing_blocks[0])


# ===================Correlation===========================================
def process_audio_correlate(indata):
     block = signal.correlate(indata,indata)
# #    block = block[len(indata)-1:len(indata)-1 + 2**15]
     return(block)

auto6=process_audio_correlate(data_trianing_blocks6[0])
auto10=process_audio_correlate(data_trianing_blocks10)

print(auto6)

# plt.plot(auto6,label="6")
# plt.plot(auto10,label="10")
# plt.show()
# =============================================================================

# ===================MFCC===========================================

MFCC6 = librosa.feature.mfcc(data_trianing_blocks6[732], n_fft=2048, hop_length=512, n_mfcc=13)

librosa.display.specshow(MFCC6, sr=sample_rate6, hop_length=512)

plt.xlabel("Time")
plt.ylable("MFCC")
plt.colorbar()
plt.show()
# =============================================================================

y = fft(data)

data2 = np.abs(y)

# print(data2)
a = np.array([1,2,3,4,5,6,7,8,9])
a.shape = (9,1)


z = np.tile(a, (325333, 1))

b= np.array([1,2,3])

z.shape = (2927997,1)

b.shape = (3,1)
k = np.append(z, b)




k.shape = (2928000,1)
# print(data2.shape)
l = np.append(data2, k,axis=1)

df = pd.DataFrame(librosa_list_new)

print(df)

# X=df.iloc[:, 0:8].values
# Y = df[8]

# print(X)
# print(Y)


# =============================================================================
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# 
# 
# #model_RF = RandomForestClassifier(n_estimators=10)
# model_LN = linear_model.LinearRegression()
# 
# #model_RF.fit(X_train, y_train)
# model_LN.fit(X_train, y_train)
# 
# #y_pred_RF = model_RF.predict(X_test)
# 
# y_pred_LN = model_LN.predict(X_test)
# 
# #print("classifier accuracy:", r2_score(y_test, y_pred_RF))
# print("classifier accuracy:", r2_score(y_test, y_pred_LN))
# 
# # =============================================================================
# # def create_model():
# #         # create model
# #         model = Sequential()
# #         model.add(Dense(100, input_shape=(9,), activation='relu'))
# #         model.add(Dense(200, activation='relu'))
# #         model.add(Dense(7, activation='relu'))
# #         # Compile model
# #         model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# #         #model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# #         #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# #         return model
# # 
# # # encode class values as integers
# # #encoder = LabelEncoder()
# # #encoder.fit(Y)
# # #encoded_Y = encoder.transform(Y)
# # # create model
# # print(y_test)
# # print(y_train)
# # 
# # y_test = to_categorical(y_test, 7)
# # y_train = to_categorical(y_train, 7)
# # 
# # Model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)
# # 
# # #y_test = to_categorical(y_test, 8)
# # #y_train = to_categorical(y_train, 8)
# # 
# # Model.fit(X_train,y_train)
# # score = Model.score(X_test, y_test)
# # 
# # print(score)
# # =============================================================================
# 
# =============================================================================
