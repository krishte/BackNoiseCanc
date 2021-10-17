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
from pydub import AudioSegment

#put noisy dataset in array
path = '/Users/krishte/Documents/Programming/Python/MachineLearning/BestFanNoise'
noisyfiles = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
      if ".wav" in file:
        #y, s = librosa.load(os.path.join(r, file), sr=8000)
        #librosa.output.write_wav(os.path.join('/Users/krishte/Documents/Programming/Python/MachineLearning/downsampled_clean_trainset', file), y, 8000)
        noisyfiles.append(os.path.join(r, file))

#print(noisyfiles[:10])
actualnoisyfiles = np.array(noisyfiles)
actualnoisyfiles = np.sort(actualnoisyfiles)

#put clean dataset in array
cleanfiles = []
path2 = '/Users/krishte/Documents/Programming/Python/MachineLearning/new_downsampled_clean_trainset'
for r, d, f in os.walk(path2):
    for file in f:
      if ".wav" in file:
        #y, s = librosa.load(os.path.join(r, file), sr=8000)    
        #librosa.output.write_wav(os.path.join('/Users/krishte/Documents/Programming/Python/MachineLearning/downsampled_clean_trainset', file), y, 8000)
        cleanfiles.append(os.path.join(r, file))
#print(cleanfiles[:10])
actualcleanfiles = np.array(cleanfiles)
actualcleanfiles = np.sort(actualcleanfiles)

rate, data = scipy.io.wavfile.read(actualnoisyfiles[100])
rate2, data2 = scipy.io.wavfile.read(actualcleanfiles[100])

print(actualnoisyfiles[100])

print(actualcleanfiles[100])

#return a random file number

x = random.randint(1,20000)

#format random wav file for graphing
spf = wave.open(actualnoisyfiles[x], 'r')
spf2 = wave.open(actualcleanfiles[x], 'r')
signal = spf.readframes(-1)
signal = np.fromstring(signal, 'Int16')
fs = spf.getframerate()
signal2 = spf2.readframes(-1)
signal2 = np.fromstring(signal2, 'Int16')

#account for sampling rate to display seconds
Time = np.linspace(0, len(signal)/fs, num=len(signal))

#using matplotlib
plt.figure()
plt.subplot(211)

#attempt at creating button
def on_click(event):
    if event.dblclick:
        audio = AudioSegment.from_wav(noisyfiles[x])
        #play(audio)

def _yes(event):
    print("working") 



plt.connect("button_press_event", on_click)
axcut = plt.axes([0.9, 0.0, 0.1, 0.075])
bcut = Button(axcut, 'YES', color='red', hovercolor='green')
bcut.on_clicked(_yes) 

#plot two graphs in subplots
plt.plot(Time, signal)
plt.grid(True)
plt.title("Noisy Speech")

plt.subplot(212)
plt.plot(Time, signal2)
plt.grid(True)
plt.title("Clean Speech")
plt.subplots_adjust(hspace= 0.5, wspace=0.5)


#class for click event
class Index(object):
    def playNoisy(self, event):
        playsound(actualnoisyfiles[x])
    def playClean(self, event):
        playsound(actualcleanfiles[x])

callback = Index()
#location of button 
axbutton = plt.axes([0.65, 0.88, 0.15, 0.075])
bplay = Button(axbutton, 'Noisy play')
bplay.color = '#5b9cf4'
bplay.hovercolor = '#2667bf'
#call play function
bplay.on_clicked(callback.playNoisy)

callback2 = Index()
axbutton2 = plt.axes([0.65, 0.42, 0.15, 0.075])
bplay2 = Button(axbutton2, 'Clean play')
bplay2.color = '#5b9cf4'
bplay2.hovercolor = '#2667bf'
bplay2.on_clicked(callback.playClean)

#display
plt.show()




STFTnoisylistreal = []
STFTnoisylistimag = []
STFTcleanlistreal = []
STFTcleanlistimag = []


for i in range(4000):
    ratetrial, datatrial = scipy.io.wavfile.read(actualnoisyfiles[i])
    f,t, STFTtrial = scipy.signal.stft(datatrial, ratetrial, 'hamming', 256, 192, 256,return_onesided = True, boundary = None, padded = False, axis = 0) 

    real = tf.math.real(STFTtrial.transpose())
    imag = tf.math.imag(STFTtrial.transpose())
    STFTnoisylistreal.append(real)
    STFTnoisylistimag.append(imag)

    ratetrial2, datatrial2 = scipy.io.wavfile.read(actualcleanfiles[i])
    f2,t2,STFTtrial2 = scipy.signal.stft(datatrial2, ratetrial2, 'hamming', 256, 192, 256,return_onesided = True, boundary = None, padded = False, axis = 0)

    real2 = tf.math.real(STFTtrial2.transpose())
    imag2 = tf.math.imag(STFTtrial2.transpose())
    STFTcleanlistreal.append(real2)
    STFTcleanlistimag.append(imag2)
       



    if (i%100==0):
      print(i) 

print("STFTdone")


STFTnoisylistreal = np.dstack(STFTnoisylistreal)
STFTcleanlistreal = np.dstack(STFTcleanlistreal)
STFTnoisylistimag = np.dstack(STFTnoisylistimag)
STFTcleanlistimag = np.dstack(STFTcleanlistimag)

STFTnoisylistreal = np.rollaxis(STFTnoisylistreal, -1)
STFTcleanlistreal = np.rollaxis(STFTcleanlistreal, -1)
STFTnoisylistimag = np.rollaxis(STFTnoisylistimag, -1)
STFTcleanlistimag = np.rollaxis(STFTcleanlistimag, -1)


inputs_real = tf.keras.Input(shape=(247,129))  # Returns an input placeholder

x = tf.keras.layers.Dense(1024, activation=tf.nn.relu)(inputs_real)

x = tf.keras.layers.Dense(1024, activation=tf.nn.relu)(x)

x = tf.keras.layers.Dense(129, activation=tf.nn.relu)(x)

model_real = tf.keras.Model(inputs=inputs_real, outputs=x, name="real_trainer")
print(model_real.summary())

inputs_imag = tf.keras.Input(shape=(247,129)) 

x = tf.keras.layers.Dense(1024, activation=tf.nn.relu)(inputs_imag)

x = tf.keras.layers.Dense(1024, activation=tf.nn.relu)(x)

x = tf.keras.layers.Dense(129, activation=tf.nn.relu)(x)

model_imag = tf.keras.Model(inputs=inputs_imag, outputs=x, name = "imag_trainer")
print(model_imag.summary())

optimizer = tf.keras.optimizers.RMSprop(0.001)

model_real.compile(loss='mean_squared_error',
              optimizer=optimizer,
          metrics=['mean_absolute_error', 'mean_squared_error'])

model_imag.compile(loss='mean_squared_error',
              optimizer=optimizer,
          metrics=['mean_absolute_error', 'mean_squared_error'])


early_stop = keras.callbacks.EarlyStopping(monitor='mean_squared_error', patience=10)

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 1 == 0: print('')
    print('.', end='')

EPOCHS = 3

history = model_real.fit(
  STFTnoisylistreal, STFTcleanlistreal,
  epochs=EPOCHS, validation_split = 0.0,
  callbacks=[early_stop, PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

history2 = model_imag.fit(
  STFTnoisylistimag, STFTcleanlistimag,
  epochs=EPOCHS, validation_split = 0.0,
  callbacks=[early_stop, PrintDot()])

hist2 = pd.DataFrame(history2.history)
hist2['epoch'] = history2.epoch
print(hist2.tail())

loss, mae, mse = model_real.evaluate(STFTnoisylistreal, STFTcleanlistreal, verbose=0)
loss2, mae2, mse2 = model_imag.evaluate(STFTnoisylistimag, STFTcleanlistimag, verbose=0)

test_predictions_real = model_real.predict(STFTnoisylistreal)
test_predictions_imag = model_imag.predict(STFTnoisylistimag)

model_real.save('realmodelnew.h5')
model_imag.save('imagmodelnew.h5')


audiofilestft = tf.complex(test_predictions_real[68].transpose(), test_predictions_imag[68].transpose())

q, audiofile = scipy.signal.istft(audiofilestft, 16000, 'hamming', 256, 192, 256 ,input_onesided = True, boundary = None)

datatrialawesome = np.around(audiofile)
datatrialawesome = np.int16(datatrialawesome)
print(datatrialawesome)
ratetrial, datatrial = scipy.io.wavfile.read(actualcleanfiles[68])
ratetrial2, datatrial2 = scipy.io.wavfile.read(actualnoisyfiles[68])
print(datatrial)
scipy.io.wavfile.write("cool.wav", 16000, datatrialawesome)
scipy.io.wavfile.write("bob.wav", 16000, datatrial)
scipy.io.wavfile.write("bob2.wav", 16000, datatrial2)

spf = wave.open("cool.wav", 'r')
spf2 = wave.open("bob.wav", 'r')
signal = spf.readframes(-1)
signal = np.fromstring(signal, 'Int16')
fs = spf.getframerate()
signal2 = spf2.readframes(-1)
signal2 = np.fromstring(signal2, 'Int16')

spf3 = wave.open("bob2.wav", 'r')
signal3 = spf3.readframes(-1)
signal3 = np.fromstring(signal3, 'Int16')


#account for sampling rate to display seconds
Time = np.linspace(0, len(signal)/fs, num=len(signal))

#using matplotlib
plt.figure()
plt.subplot(312)
plt.plot(Time, signal)
plt.grid(True)
plt.title("Neural Network Clean Speech")

plt.subplot(313)
plt.plot(Time, signal2)
plt.grid(True)
plt.title("Clean Speech")
plt.subplots_adjust(hspace= 0.5, wspace=0.5)

plt.subplot(311)
plt.plot(Time, signal2)
plt.grid(True)
plt.title("Noisy Speech")
plt.subplots_adjust(hspace= 0.5, wspace=0.5)


plt.show()

os.remove("cool.wav")
os.remove("bob.wav")
os.remove("bob2.wav")
