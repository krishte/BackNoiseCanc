
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




#put noisy dataset in array
path = '/Users/krishte/Documents/Programming/Python/MachineLearning/noisy_trainset_wav'
noisyfiles = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        #y, s = librosa.load(os.path.join(r, file), sr=8000)
        #librosa.output.write_wav(os.path.join('/Users/krishte/Documents/Programming/Python/MachineLearning/downsampled_clean_trainset', file), y, 8000)
        noisyfiles.append(os.path.join(r, file))

#print(noisyfiles[:10])
actualnoisyfiles = np.array(noisyfiles)
#print(actualnoisyfiles)

#put clean dataset in array
cleanfiles = []
path2 = '/Users/krishte/Documents/Programming/Python/MachineLearning/clean_trainset_wav'
for r, d, f in os.walk(path2):
    for file in f:
        if ".wav" in file:
        #y, s = librosa.load(os.path.join(r, file), sr=8000)    
        #librosa.output.write_wav(os.path.join('/Users/krishte/Documents/Programming/Python/MachineLearning/downsampled_clean_trainset', file), y, 8000)
            cleanfiles.append(os.path.join(r, file))
#print(cleanfiles[:10])
actualcleanfiles = np.array(cleanfiles)
#print(actualcleanfiles)




for z in range(len(actualcleanfiles)):
    if ".wav" in actualcleanfiles[z]:
        rate, data = scipy.io.wavfile.read(actualcleanfiles[z])

        myaudio = AudioSegment.from_file(actualcleanfiles[z] , "wav") 
        chunk_length_ms = 1000 # pydub calculates in millisec
        chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec
        
        for i in range(len(data) // 16000):
            chunk_name =  "226_"+ str(z) + "_" +"chunk{0}.wav".format(i)
            print("exporting", chunk_name)
            chunks[i].export(chunk_name, format="wav")

shortenedfiles = []
path2 = '/Users/krishte/Documents/Programming/Python/MachineLearning/downsampled_clean_trainset'
for r, d, f in os.walk(path2):
    for file in f:
        if '.wav' in file:
        #y, s = librosa.load(os.path.join(r, file), sr=8000)    
        #librosa.output.write_wav(os.path.join('/Users/krishte/Documents/Programming/Python/MachineLearning/downsampled_clean_trainset', file), y, 8000)
            shortenedfiles.append(os.path.join(r, file))
#print(cleanfiles[:10])
shortenedfiles = np.array(shortenedfiles)
print(shortenedfiles)

for i in range(len(shortenedfiles)):
    rate, data = scipy.io.wavfile.read(shortenedfiles[i])
    print(rate, len(data))
