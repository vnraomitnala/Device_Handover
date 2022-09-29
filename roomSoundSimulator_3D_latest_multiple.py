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
import os
import math


with open ('0.8_.pickle', 'rb') as f:
    datasetMSC = pickle.load(f) 

with open ('linear_move.pickle', 'rb') as f:
    datasetDistance = pickle.load(f)     

with open ('locus_positions_6.pickle', 'rb') as f:
    locus1 = pickle.load(f)   
    
   

x_without_smooth = datasetMSC["microphone_without_smooth"]
x_with_smooth = datasetMSC["microphone_with_smooth"]

source_locs = [ [0 +i*0.07, 3,1] for i in range(101) ]



confRoom_source_locs = [ [0.5 +i*0.05, 1, 1.4] for i in range(115) ]

standardRoom_source_locs2 = [ [0.5 +i*0.05,1, 1.4] for i in range(111) ]

#locus1 = standardRoom_source_locs2



y = []

for i in range(len(locus1)):
    y.append(locus1[i][0])
    
df = pd.DataFrame(datasetMSC)

fig, axs = plt.subplots(
        nrows=5, ncols=1, sharex=True, sharey=False, 
        gridspec_kw={'height_ratios':[1,1,1,3,3]}
        
        )

axs[0].step(y, (datasetDistance['mic']), c ="blue", label='I')
axs[0].grid()
axs[0].legend()
#plt.ylabel('$y$ (transition)')


axs[1].step(y,x_with_smooth, 'r' , label='S')
axs[1].grid()
axs[1].legend()
axs[1].set_ylabel('$y$ (transition)')

axs[2].step(y,x_without_smooth, 'r' , label='NS')
axs[2].grid()
axs[2].legend()


axs[2].set_xlabel('$x$ (distance)')
#plt.ylabel('$y$ (transition)')

#plt.savefig('0.6_CR_1.pdf')

axs[3].step(y, (datasetMSC['mic1_coherence_with_smooth']), c ="blue", label='Mic1 Coh. S')
axs[3].step(y, (datasetMSC['mic2_coherence_with_smooth']), c ="orange", label='Mic2 Coh. S')
axs[3].grid()
axs[3].legend()
#plt.ylabel('$y$ (coherence)')


axs[4].step(y, (df['mic1_coherence_without_smooth']), c ="blue", label='Mic1 Coh. NS', alpha=0.8)
axs[4].step(y, (df['mic2_coherence_without_smooth']), c ="orange", label='Mic2 Coh. NS', alpha=0.6)
axs[4].grid()
axs[4].legend()


axs[4].set_xlabel('$x$ (distance)')
axs[4].set_ylabel('                            $y$ (coherence)')

plt.savefig('0.8_locus4.pdf')
plt.show()

ideal = [int(x) for x in datasetDistance['mic']]
xS = [int(x) for x in x_with_smooth]
xNS = [int(x) for x in x_without_smooth]


sq_error_without_smooth = np.subtract(ideal, xNS) ** 2
mse_without_smooth = sq_error_without_smooth.mean()
print("mse_without_smooth: ", mse_without_smooth)

wtd_mse_without_smooth = []

xx = np.abs(np.subtract(datasetDistance['mic1-distance'], datasetDistance['mic2-distance']))

for x, y in zip(sq_error_without_smooth, xx):
    wtd_mse_without_smooth.append(x * y/np.sum(xx))

wtd_mse_without_smooth_mean = sum(wtd_mse_without_smooth)

print('wtd_mse_without_smooth: ', wtd_mse_without_smooth_mean)


sq_error_with_smooth = np.subtract(ideal, xS) ** 2
mse_with_smooth = sq_error_with_smooth.mean()
print("mse_with_smooth: ", mse_with_smooth)

wtd_mse_with_smooth = []



for x, y in zip(sq_error_with_smooth, xx):
    wtd_mse_with_smooth.append(x * y/np.sum(xx))

wtd_mse_with_smooth_mean = sum(wtd_mse_with_smooth)
print('wtd_mse_with_smooth: ', wtd_mse_with_smooth_mean)

def differences(a, b):
    if len(a) != len(b):
        raise ValueError("Lists of different length.")
    return sum(i != j for i, j in zip(a, b))

diff_I_xNS = differences(ideal, xNS)
diff_I_xS =  differences(ideal, xS) 

if diff_I_xNS > 0:
    diff_I_xNS = diff_I_xNS-1
    
if diff_I_xS > 0:
    diff_I_xS = diff_I_xS -1
    
    
              
print( "No of transitions different from Ideal-without_smooth: ", diff_I_xNS*2)
print( "No of transitions different from Ideal-with_smooth: ", diff_I_xS *2)


