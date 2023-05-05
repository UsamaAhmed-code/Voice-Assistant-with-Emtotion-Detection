

from neuralintents import GenericAssistant
import speech_recognition
import pyttsx3 as tts
import sys 



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import pandas as pd
import glob
from sklearn.metrics import confusion_matrix
import IPython.display as ipd # to play audio in the notebook
import os
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Keras
import keras
from keras import regularizers
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Embedding, LSTM
from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.utils import np_utils, to_categorical
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
from keras import losses, models, optimizers
from keras.activations import relu, softmax
from keras.layers import (Convolution2D, GlobalAveragePooling2D, BatchNormalization, Flatten, Dropout,
                          GlobalMaxPool2D, MaxPool2D, concatenate, Activation, Input, Dense)
# sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Other  
from tqdm import tqdm, tqdm_pandas
import scipy
from scipy.stats import skew
import librosa
import librosa.display
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import pandas as pd
import seaborn as sns
import glob 
import os
import sys
import IPython.display as ipd  # To play sound in the notebook
import warnings
# ignore warnings 
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import pickle





import sounddevice as sd
from scipy.io.wavfile import write

fs = 44100  # Sample rate
seconds = 6  # Duration of recording
print("start speaking")
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
sd.wait()  # Wait until recording is finished
write('C:/Users/usama/VA-with-ER/speech_and_text_emotion_recognition/prediction_audio_files/output.wav', fs, myrecording)  # Save as WAV file 





# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def prepare_prediction_audioPaths(folderPath):
  df = pd.DataFrame(columns=['path'])
  paths = []
  for fileName in os.listdir(folderPath):
    paths.append(folderPath+fileName)
  df['path'] = paths
  print(df)
  return df

# Extracting the MFCC feature as an image (Matrix format). 
def extract_data_from_audios(df_paths, n, aug, mfcc):
    X = np.empty(shape=(len(df_paths), n, 216, 1))
    input_length = sampling_rate * audio_duration
    
    cnt = 0
    # print('df_paths: ', df_paths)
    for fname in df_paths:
        print('fname: ', fname)
        file_path = fname
        print('file_path: ', file_path)
        data, _ = librosa.load(file_path, sr=sampling_rate
                               ,duration=2.5
                               ,offset=0.5
                              )
        # Random offset / Padding
        if len(data) > input_length:
            max_offset = len(data) - input_length
            offset = np.random.randint(max_offset)
            data = data[offset:(input_length+offset)]
        else:
            if input_length > len(data):
                max_offset = input_length - len(data)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            data = np.pad(data, (offset, int(input_length) - len(data) - offset), "constant")

            # Augmentation? 
        if aug == 1:
            data = speedNpitch(data)
        
        # which feature?
        if mfcc == 1:
            # MFCC extraction 
            MFCC = librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=n_mfcc)
            MFCC = np.expand_dims(MFCC, axis=-1)
            X[cnt,] = MFCC
            
        else:
            # Log-melspectogram
            melspec = librosa.feature.melspectrogram(data, n_mels = n_melspec)   
            logspec = librosa.amplitude_to_db(melspec)
            logspec = np.expand_dims(logspec, axis=-1)
            X[cnt,] = logspec
            
        cnt += 1

    return X
def normalize_data(X_pred):
  with open(dataFilePath + "mean_std.pickle", 'rb') as file:
    mean_std = pickle.load(file)
  mean = mean_std[0]
  std= mean_std[1]
  X_pred = (X_pred - mean)/std
  return X_pred

def _load_model_(filePath, fileName):
  # Loading json and model architecture -----------------------
  with open(filePath + fileName + ".json", 'r') as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
  # Load weights into new model
  loaded_model.load_weights(filePath + fileName +".h5")
  print("--> Loaded model from disk")
  
  # Keras optimiser -------------------------------------------
  opt = tf.keras.optimizers.RMSprop(lr=0.00001)
  loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
  """score = loaded_model.evaluate(X_test, y_test, verbose=0)
  print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))"""
  return loaded_model

def model_predict(model, X_test):
  preds = model.predict(X_test, batch_size=16, verbose=1)
  preds = preds.argmax(axis=1)

  with open(dataFilePath + "LableEncoder.pickle", 'rb') as file: 
    LE = pickle.load(file)
  # Appending the labels
  preds = preds.astype(int).flatten()
  preds = (LE.inverse_transform((preds)))
  # preds = pd.DataFrame({'predictedvalues': preds})
  return preds[0]
   

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


rootPath = "C:/Users/usama/VA-with-ER/speech_and_text_emotion_recognition/"
dataFilePath = rootPath + "saved_data/"
predFolderPath = rootPath + "prediction_audio_files/"

sampling_rate = 44100
audio_duration = 2.5
n_mfcc = 30

modelFileName = "mfcc_model"

# =========================== Extracting audio paths into dataframe
df = prepare_prediction_audioPaths(predFolderPath)
print('----------------------------------------')
print(df['path'])
print('--------------------------------------------------------')
# =========================== Extracting data from audio files
X_pred = extract_data_from_audios(df['path'], n = n_mfcc, aug = 0, mfcc = 1)
print(X_pred)
# =========================== Normalizing the data
X_pred = normalize_data(X_pred)
# =========================== Loading the Model
model = _load_model_(dataFilePath, modelFileName)
print("after model loading")
# =========================== Predictions
preds = model_predict(model, X_pred)
print("after pred")
# =========================== Save Predictions
# preds.to_csv(predFolderPath + "predictions.csv", index=False)
print(preds)




recognizer = speech_recognition.Recognizer()

speaker = tts.init()
speaker.setProperty('rate', 150)

todo_list = ['go shopping', 'Clean Room', 'Record Video']

sql = todo_list

def create_note():
    global recognizer

    speaker.say(" what you want to write onto your note?")
    speaker.runAndWait()

    done = False

    while not done:
        try:

            with speech_recognition.Microphone() as mic:
                recognizer.adjust_for_ambient_noise(mic, duration=0.2)
                audio = recognizer.listen(mic)
              
                
                note = recognizer.recognize_google(audio)
                note = note.lower()

                speaker.say("Choose a filename")
                speaker.runAndWait()

                recognizer.adjust_for_ambient_noise(mic, duration=0.2)
                audio = recognizer.listen(mic)

                filename = recognizer.recognize_google(audio)
                filename = filename.lower()

                with open(filename, 'w') as f:
                     f.write(note)
                     done = True
                     speaker.say("I successfully created the note {filename}")
                     speaker.runAndWait()

        except speech_recognition.UnknownValueError:
                    recognizer = speech_recognition.Recognizer()
                    speaker.say("I did not understand you! Please try again")
                    speaker.runAndWait()   

def add_todo():
     global recognizer

     speaker.say("what todo you want to add?")
     speaker.runAndWait()

     done = False

     while not done:
          try:
               with speech_recognition.Microphone() as mic:
                    recognizer.adjust_for_ambient_noise(mic, duration=0.2)
                    audio = recognizer.listen(mic)

                    item = recognizer.recognize_google(audio)
                    item = item.lower()

                    todo_list.append(item)
                    done = True

                    speaker.say("I added {item} to the to do list!")
                    speaker.runAndWait()

          except speech_recognition.UnknownValueError:
               recognizer = speech_recognition.Recognizer()
               speaker.say("I did not understand. Please try again!")
               speaker.runAndWait()

def show_todo():
     
     speaker.say("the items on your todo list are the following")
     for item in todo_list:
          speaker.say(item)
     speaker.runAndWait()

def hello():
     speaker.say("Hello, What can I do for you?")
     speaker.runAndWait()



def quit():
     if preds == "male_sad" :
    
      speaker.say("good bye . I can feel you are sad. be happy man. don't worry. have a nice day ")
      speaker.runAndWait()
      sys.exit(0)

     elif preds == "female_sad" :
    
       speaker.say("good bye . I can feel you are sad. cheerup lady . don't worry. have a nice day ")
       speaker.runAndWait()
       sys.exit(0) 

     elif preds == "male_happy" :
    
       speaker.say("good bye . I can feel you are happy. stay blessed Man. ")
       speaker.runAndWait()
       sys.exit(0) 

     elif preds == "female_happy" :
    
       speaker.say("good bye . I can feel you are very happy today. stay blessed lady . ")
       speaker.runAndWait()
       sys.exit(0) 

      
     elif preds == "male_angry" :
    
       speaker.say("good bye . I can feel you are angry. be calm and relax man, life is short be happy")
       speaker.runAndWait()
       sys.exit(0)

     elif preds == "female_angry" :
    
       speaker.say("good bye . I can feel you are very angry . be calm and relax, life is short be happy girl ")
       speaker.runAndWait()
       sys.exit(0) 

     elif preds == "male_fear" :
    
       speaker.say("good bye . I can feel you are very afraid . i am here to help you. don't worry ")
       speaker.runAndWait()
       sys.exit(0)

     elif preds == "female_fear" :
    
       speaker.say("good bye . I can feel you are afraid of something. i am here to help you. don't worry ")
       speaker.runAndWait()
       sys.exit(0)  

     elif preds == "male_calm" :
    
       speaker.say("good bye . I can feel you are very clam. be like this. have a good day ")
       speaker.runAndWait()
       sys.exit(0) 

     elif preds == "female_calm" :
    
       speaker.say("good bye . I can feel you are  clam today. be like this. have a good day ")
       speaker.runAndWait()
       sys.exit(0)  

     elif preds == "male_disgust" :
    
       speaker.say("good bye . I can feel you are feeling disgust. be relax ")
       speaker.runAndWait()
       sys.exit(0) 

     elif preds == "female_disgust" :
    
       speaker.say("good bye . I can feel you are feeling disgust today. be relax ")
       speaker.runAndWait()
       sys.exit(0) 
  
     elif preds == "male_neutral" :
    
       speaker.say("good bye . I can feel you are looking neutral today. good to hear ")
       speaker.runAndWait()
       sys.exit(0) 

     elif preds == "female_neutral" :
    
       speaker.say("good bye . I can feel you are looking neutral . good to know ")
       speaker.runAndWait()
       sys.exit(0) 

     elif preds == "male_surprise" :
    
       speaker.say("good bye . I can feel you are looking suprised today. be relax ")
       speaker.runAndWait()
       sys.exit(0) 
      
      
     elif preds == "female_surprise" :
    
       speaker.say("good bye . I can feel you are feeling suprised . be relax ")
       speaker.runAndWait()
       sys.exit(0) 



       

#     elif mean_pitch <= 0.020 :  
     
 #     speaker.say("good bye have a nice day. I can feel you are sad. be happy and enjoy")
  #    speaker.runAndWait()
   #   sys.exit(0)

     else:
         speaker.say("good bye have a nice day.")
         speaker.runAndWait()
         sys.exit(0)

mapping={
     "greeting": hello,
     "create_note": create_note,
     "add_todo": add_todo,
     "show_todo": show_todo,
     "exit": quit
}    

            
assistant = GenericAssistant('index.json', intent_methods=mapping)
assistant.train_model()

assistant.save_model()
assistant.load_model()



import librosa

      # Load audio file
y, sr = librosa.load('C:/Users/usama/VA-with-ER/speech_and_text_emotion_recognition/output.wav')

# Calculate pitch
pitch, _ = librosa.core.pitch.piptrack(y=y, sr=sr)

# Get the mean pitch
mean_pitch = librosa.core.pitch_tuning(pitch)

# Print the mean pitch
print(mean_pitch)
 

while True:
     try:
        
          with speech_recognition.Microphone() as mic:
               
               recognizer.adjust_for_ambient_noise(mic, duration=0.2)
               audio = recognizer.listen(mic)

               message = recognizer.recognize_google(audio)
               message = message.lower()
           
          assistant.request(message)
     except speech_recognition.UnknownValueError:
          recognizer = speech_recognition.Recognizer()



