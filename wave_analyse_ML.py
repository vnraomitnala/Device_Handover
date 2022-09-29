# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 09:21:27 2021

@author: Vijaya
"""

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
import csv
import pickle
import scipy.stats
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
from matplotlib.backends.backend_pdf import PdfPages


samplerate, data = scipy.io.wavfile.read('C:\\Users\\Vijaya\\PhD\\MicrophoneArray_Dataset\\tabletop11\\tabletop11_.wav')

print(data.shape)

y = fft(data)

data2 = np.abs(y)

print(data2)
a = np.array([1,2,3,4,5,6,7,8,9])
a.shape = (9,1)

x = np.arange(1, 2928000, 1).T

z = np.tile(a, (325333, 1))

b= np.array([1,2,3])

z.shape = (2927997,1)

b.shape = (3,1)
k = np.append(z, b)

k.shape = (2928000,1)


print(data2.shape)
l = np.append(data2, k,axis=1)


df = pd.DataFrame(l)

print(df)

X = df.iloc[:, 0:8].values
Y = df[8]

print(X)
print(Y)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


model = RandomForestClassifier(n_estimators=10)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("classifier accuracy:", accuracy_score(y_test, y_pred))


