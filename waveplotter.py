import os
import random
import sys
import wave

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import soundfile as sf
import tensorflow as tf
from matplotlib.widgets import Button
from playsound import playsound
from scipy import signal
from scipy.io import wavfile
from tensorflow import keras
from pydub import effects

#put noisy dataset in array
path = '/Users/krishte/Documents/Programming/Python/MachineLearning/BestFanNoise'
noisyfiles = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
      if ".wav" in file:
        noisyfiles.append(os.path.join(r, file))


actualnoisyfiles = np.array(noisyfiles)
actualnoisyfiles = np.sort(actualnoisyfiles)
print(actualnoisyfiles)

#put clean dataset in array
cleanfiles = []
path2 = '/Users/krishte/Documents/Programming/Python/MachineLearning/new_downsampled_clean_trainset'
for r, d, f in os.walk(path2):
    for file in f:
      if ".wav" in file:
        cleanfiles.append(os.path.join(r, file))
actualcleanfiles = np.array(cleanfiles)
actualcleanfiles = np.sort(actualcleanfiles)


model_real = tf.keras.models.load_model('realmodelnew.h5')
model_imag = tf.keras.models.load_model('imagmodelnew.h5')

print(model_real.summary())
print(model_imag.summary())

optimizer = tf.keras.optimizers.RMSprop(0.001)

model_real.compile(loss='mean_squared_error',
              optimizer=optimizer,
          metrics=['mean_absolute_error', 'mean_squared_error'])

model_imag.compile(loss='mean_squared_error',
              optimizer=optimizer,
          metrics=['mean_absolute_error', 'mean_squared_error'])

    

STFTnoisylistreal = []
STFTnoisylistimag = []
#return a random file number
plt.figure(figsize=(12,8))
x= int(input("What speech files: "))
counter = 0

for i in range(x, x+7):
    ratetrial, datatrial = scipy.io.wavfile.read(actualnoisyfiles[i])
    f,t, STFTtrial = scipy.signal.stft(datatrial, ratetrial, 'hamming', 256, 192, 256,return_onesided = True, boundary = None, padded = False, axis = 0) 
    real = tf.math.real(STFTtrial.transpose())
    imag = tf.math.imag(STFTtrial.transpose())
    STFTnoisylistreal.append(real)
    STFTnoisylistimag.append(imag)

ratetrial2, datatrial2 = scipy.io.wavfile.read(actualnoisyfiles[127])
#scipy.io.wavfile.write("bobbity.wav", ratetrial2, datatrial2)

STFTnoisylistreal = np.dstack(STFTnoisylistreal)
STFTnoisylistimag = np.dstack(STFTnoisylistimag)

STFTnoisylistreal = np.rollaxis(STFTnoisylistreal, -1)
STFTnoisylistimag = np.rollaxis(STFTnoisylistimag, -1)

test_predictions_real = model_real.predict(STFTnoisylistreal)
test_predictions_imag = model_imag.predict(STFTnoisylistimag)

for i in range(7):
    audiofilestft = tf.complex(test_predictions_real[i].transpose(), test_predictions_imag[i].transpose())
    q, audiofile = scipy.signal.istft(audiofilestft, 16000, 'hamming', 256, 192, 256 ,input_onesided = True, boundary = None)
    datatrialawesome = np.around(audiofile)
    datatrialawesome = np.int16(datatrialawesome)
    scipy.io.wavfile.write("cool" + str(i) + ".wav", 16000, datatrialawesome)


for i in range(x, x+7):


    spf = wave.open(actualnoisyfiles[i], 'r')
    spf2 = wave.open(actualcleanfiles[i], 'r')
    spf3 = wave.open("cool" + str(i-x) + ".wav", 'r')
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, 'Int16')
    fs = spf.getframerate()
    signal2 = spf2.readframes(-1)
    signal2 = np.fromstring(signal2, 'Int16')
    signal3 = spf3.readframes(-1)
    signal3 = np.fromstring(signal3, 'Int16')
    Time = np.linspace(0, len(signal)/fs, num=len(signal))


    plt.subplot(8,3,counter+1)
    plt.plot(Time, signal)
    plt.ylim(-15000, 15000)
    plt.grid(True)
    plt.title("Noisy Speech" + str(i))

    plt.subplot(8, 3, counter+2)
    plt.plot(Time, signal2)
    plt.ylim(-15000, 15000)
    plt.grid(True)
    plt.title("Clean Speech" + str(i))
    plt.subplots_adjust(hspace= 2, wspace=0.5)

    plt.subplot(8, 3, counter+3)
    plt.plot(Time, signal3)
    plt.ylim(-15000, 15000)
    plt.grid(True)
    plt.title("NN Speech" + str(i))


    counter += 3

plt.tight_layout()
plt.savefig("noiseplots.png")
#plt.show()

for i in range(7):
   os.remove("cool" + str(i) + ".wav")


