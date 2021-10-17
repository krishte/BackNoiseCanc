

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
from pydub import AudioSegment
from pydub.utils import make_chunks

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

#y, s = librosa.load('patti.wav', sr=16000) # Downsample 44.1kHz to 8kHz
#librosa.output.write_wav("patti.wav", y, 16000)



# add name of bobbity.wav

f = open("/Users/krishte/Documents/Programming/Python/MachineLearning/bob.txt", 'r')
contents = f.read()

rate, data = scipy.io.wavfile.read(contents)

print(rate, data)


myaudio = AudioSegment.from_file(contents, "wav") 
chunk_length_ms = 1000 # pydub calculates in millisec
chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

for i in range(len(data) // 16000):
    chunk_name =  "stuff_" + "chunk{0}.wav".format(i)
    print("exporting", chunk_name)
    chunks[i].export(chunk_name, format="wav")

STFTnoisylistreal = []
STFTnoisylistimag = []
for i in range(len(data)//16000):
    ratetrial, datatrial = scipy.io.wavfile.read("stuff_chunk" + str(i) + ".wav")
    f,t, STFTtrial = scipy.signal.stft(datatrial, ratetrial, 'hamming', 256, 192, 256,return_onesided = True, boundary = None, padded = False, axis = 0) 
    real = tf.math.real(STFTtrial.transpose())
    imag = tf.math.imag(STFTtrial.transpose())
    STFTnoisylistreal.append(real)
    STFTnoisylistimag.append(imag)
    


STFTnoisylistreal = np.dstack(STFTnoisylistreal)
STFTnoisylistimag = np.dstack(STFTnoisylistimag)


STFTnoisylistreal = np.rollaxis(STFTnoisylistreal, -1)
STFTnoisylistimag = np.rollaxis(STFTnoisylistimag, -1)


test_predictions_real = model_real.predict(STFTnoisylistreal)
test_predictions_imag = model_imag.predict(STFTnoisylistimag)

audiofilestft = tf.complex(test_predictions_real[0].transpose(), test_predictions_imag[0].transpose())

q, audiofile = scipy.signal.istft(audiofilestft, 16000, 'hamming', 256, 192, 256 ,input_onesided = True, boundary = None)

datatrialawesome = np.around(audiofile)
datatrialawesome = np.int16(datatrialawesome)
scipy.io.wavfile.write("result.wav", 16000, datatrialawesome)

""" spf = wave.open("cool.wav", 'r')
spf2 = wave.open("patti_chunk2.wav", 'r')
signal = spf.readframes(-1)
signal = np.fromstring(signal, 'Int16')
fs = spf.getframerate()
signal2 = spf2.readframes(-1)
signal2 = np.fromstring(signal2, 'Int16')

Time = np.linspace(0, len(signal)/fs, num=len(signal))

#using matplotlib
plt.figure()
plt.subplot(211)
plt.plot(Time, signal)
plt.grid(True)
plt.title("Neural Network Clean Speech")

plt.subplot(212)
plt.plot(Time, signal2)
plt.grid(True)
plt.title("Noisy Speech")
plt.subplots_adjust(hspace= 0.5, wspace=0.5)

plt.show()

 """



#record audio
#split audiofile into 1 second segments
#train neural network on 1 second segments
#piece one second segments together
