
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
from random import randint
import wave



cleanfiles = []
filelist = []
path2 = '/Users/krishte/Documents/Programming/Python/MachineLearning/new_downsampled_clean_trainset'
for r, d, f in os.walk(path2):
    for file in f:
      if ".wav" in file:
        #y, s = librosa.load(os.path.join(r, file), sr=8000)    
        #librosa.output.write_wav(os.path.join('/Users/krishte/Documents/Programming/Python/MachineLearning/downsampled_clean_trainset', file), y, 8000)
        cleanfiles.append(os.path.join(r, file))
        filelist.append(file)
#print(cleanfiles[:10])
actualcleanfiles = np.array(cleanfiles)
actualcleanfiles = np.sort(actualcleanfiles)
filelist = np.sort(filelist)



# newAudio = AudioSegment.from_wav("fancool.wav")
# newAudio = newAudio[2000:3000]
# newAudio.export("/Users/krishte/Documents/Programming/Python/MachineLearning/cool.wav", format="wav")





""" for i in range(len(actualcleanfiles)):
    sound1 = AudioSegment.from_file(actualcleanfiles[i])
    sound2 = AudioSegment.from_file("/Users/krishte/Documents/Programming/Python/MachineLearning/cool0.wav")

    combined = sound1.overlay(sound2)

    combined.export("/Users/krishte/Documents/Programming/Python/MachineLearning/Fannoise/noisy" + filelist[i] , format='wav')
    if i % 100 == 0:
        print(i) """


for i in range(5000):
  ratetrial, datatrial = scipy.io.wavfile.read("/Users/krishte/Documents/Programming/Python/MachineLearning/cool.wav")
  ratetrial2, datatrial2 = scipy.io.wavfile.read(actualcleanfiles[i])

  result =  datatrial[:,0] +  datatrial2
  scipy.io.wavfile.write("/Users/krishte/Documents/Programming/Python/MachineLearning/BestFanNoise/" + filelist[i], 16000, result)
  if i % 100 == 0:
    print(i)









noisyfiles = []
path3 = '/Users/krishte/Documents/Programming/Python/MachineLearning/NewFannoise'
for r, d, f in os.walk(path3):
    for file in f:
      if ".wav" in file:
        #y, s = librosa.load(os.path.join(r, file), sr=8000)    
        #librosa.output.write_wav(os.path.join('/Users/krishte/Documents/Programming/Python/MachineLearning/downsampled_clean_trainset', file), y, 8000)
        noisyfiles.append(os.path.join(r, file))
#print(cleanfiles[:10])
actualnoisyfiles = np.array(noisyfiles)
actualnoisyfiles = np.sort(actualnoisyfiles)

ratetrial, datatrial = scipy.io.wavfile.read(actualnoisyfiles[1])

