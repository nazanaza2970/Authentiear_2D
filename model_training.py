#file directory should have unzipped data in folders 'P01', 'P02' etc.
#persons includes all the folder names

# change the hyperparamters and filenames (base_model, siamese_model, loss_curve) before running

file_directory = "/content/"
persons = ["P01","P02","P03","P05"]

#pre-process parameters
order = 2
lowcut = 10
highcut = 200
fs = 500
s = 5
ov = 0.2

#training_set parameters
pos = 500
neg = 500

#training_parameters
epochs = 20
batch_size = 5
validation_split = 0.3


from keras.layers import Input, Dense, InputLayer, Conv2D, MaxPooling2D, UpSampling2D, InputLayer, Concatenate, Flatten, Reshape, Lambda, Embedding, dot
from keras.models import Model, load_model, Sequential
import matplotlib.pyplot as plt
import keras.backend as K
from sklearn.model_selection import train_test_split
import os, sys
import glob
import tensorflow as tf
from tensorflow.keras.utils import plot_model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks,butter, sosfiltfilt, sosfreqz
from sklearn.preprocessing import RobustScaler
from scipy import signal
from PIL import Image

# import librosa
from tqdm import tqdm
import random

def window_data(data, fs, s, ov):
    """
    Windows a 1d np array of timeseries data into s second frames with ov percentage of overlap.
    Needs to be applied along every column

    Args:
        data: 1d np array of timeseries data.
        fs: Sampling rate in Hz.
        s: Window length in seconds.
        ov: Overlap rate as decimal point.

    Returns:
        2d np array of windowed data.
    """

    # Calculate the window length in samples.
    window_length = int(fs * s)

    # Calculate the overlap length in samples.
    overlap_length = int(window_length * ov)

    # Calculate the number of windows
    num_windows = int(np.ceil((len(data) - window_length) / (window_length - overlap_length))) + 1

    # Initialize the output array.
    windowed_data = np.zeros((num_windows, window_length))

    # Loop over the data and window it.
    for i in range(num_windows):
        start_index = i * (window_length - overlap_length)
        end_index = min(start_index + window_length, len(data))
        windowed_data[i, :end_index - start_index] = data[start_index:end_index]

    return windowed_data

def pre_process(file_dir,person,order,lowcut,highcut,fs,s,ov):

  '''
  Args:
    file_dir: directory of all unzipped files
    person: person identifier. e.g. "P05"
    order: bpass filter order
    lowcut: low freq threshold for bpass
    highcut: high freq threshold for bpass
    fs: sampling frequency
    s: windowing time in second
    ov: overlap rate in decimal point
  Returns:
    windows of preprocessed data. (12 2D np arrays in a list, each 2D array has 1D windows from one column in the csv)
  '''

  # scan and list all files
  all_files = glob.glob(file_dir+"/**/*.csv",recursive=True)
  target_files = [x for x in all_files if person in x]
  print(target_files)
  data_l = [x for x in target_files if "ACC1" in x]
  data_r = [x for x in target_files if "ACC2" in x]

  # read left and right data
# read left and right data
  L=[]
  R=[]
  s_rate = str(int(1000/fs))+"ms"

  for idx in range(len(data_l)):
      L_=pd.read_csv(data_l[idx],names = ["timestamp","Ax","Ay","Az","Gx","Gy","Gz"])
      R_=pd.read_csv(data_r[idx],names = ["timestamp","Ax","Ay","Az","Gx","Gy","Gz"])

      # assign timestamp as index
      L_['timestamp'] = pd.to_datetime(L_['timestamp'], unit='ms')
      R_['timestamp'] = pd.to_datetime(R_['timestamp'], unit='ms')
      L_ = L_.set_index('timestamp')
      R_ = R_.set_index('timestamp')

      # resample files from both sensors to match up the values
      #https://stackoverflow.com/questions/17001389/pandas-resample-documentation
      L_=L_.resample(s_rate).mean()
      R_=R_.resample(s_rate).mean()

      # interpolate
      L_ = L_.interpolate(method="linear",axis=0)
      R_ = R_.interpolate(method="linear",axis=0)

      L.append(L_)
      R.append(R_)

  L = pd.concat(L)
  R = pd.concat(R)

  #pre-process (bandpass filter, windowing)
  data = []

  sos = butter(N=order, Wn=[lowcut,highcut], btype='bandpass', fs=fs, output='sos', analog=False)
  transformer = RobustScaler()

  for col in L.columns:
      signal = sosfiltfilt(sos, L[col])
      # signal = transformer.fit_transform(signal)
      data.append(window_data(np.array(signal),fs,s,ov))
  for col in R.columns:
      signal = sosfiltfilt(sos, R[col])
      # signal = transformer.fit_transform(signal)
      data.append(window_data(np.array(signal),fs,s,ov))

  return data

def generate_image_2(data, fs=500, window_size=0.1, overlap=0.5, target_size=(64, 64)):

  num_channels = 12
  num_windows = data[0].shape[0]
  all_images = []

  nperseg = int(window_size * fs)
  noverlap = int(nperseg * overlap)

  for i in tqdm(range(data[0].shape[0])):
  # for i in tqdm(range(5)):
    combined_image = []
    for j in range(12):
      # p5_data[j][i]
      f, t, Sxx = signal.spectrogram(data[j][i], 500, window='hamming', nperseg=nperseg, noverlap=overlap)
      Sxx_dB = 10 * np.log10(Sxx)
      # Normalize the dB values to 0-255 for grayscale image
      Sxx_norm = (Sxx_dB - np.min(Sxx_dB)) / (np.max(Sxx_dB) - np.min(Sxx_dB)) * 255
      # Convert to uint8
      Sxx_img = Sxx_norm.astype(np.uint8)
      # Resize image to target size
      img = Image.fromarray(Sxx_img)
      img_resized = img.resize(target_size)
      # Add to combined image
      combined_image.append(np.array(img_resized))
      # print(len(combined_image))
    # Stack images along the last axis to create a 12-channel image
    combined_image = np.stack(combined_image, axis=2)
    # print(combined_image.shape)
    # save images to master list
    all_images.append(combined_image)

  return all_images

def get_pairs(lists, pos, neg):
  X1 = []
  X2 = []
  y = []

  for idx in range(pos+neg):
   #for positive, take from same list
    list_a = lists[random.randint(0,len(lists)-1)]

    X1.append(list_a[random.randint(0,len(list_a)-1)])
    X2.append(list_a[random.randint(0,len(list_a)-1)])
    y.append(0)

   #for negative, take from different lists
    ind_1 = random.randint(0,len(lists)-1)
    ind_2 = random.randint(0,len(lists)-1)
    while ind_1 == ind_2:
      ind_2 = random.randint(0,len(lists)-1)
    
    list_a = lists[ind_1]
    list_b = lists[ind_2]

    X1.append(list_a[random.randint(0,len(list_a)-1)])
    X2.append(list_b[random.randint(0,len(list_b)-1)])
    y.append(1)
  
  return X1,X2, y

#Get the base model for siamese network
input_layer = Input((64, 64, 12))
layer1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
layer2 = MaxPooling2D((2, 2), padding='same')(layer1)
layer3 = Conv2D(32, (3, 3), activation='relu', padding='same')(layer2)
layer4 = MaxPooling2D((2, 2), padding='same')(layer3)
layer5 = Flatten()(layer4)
embeddings = Dense(16, activation=None)(layer5)
norm_embeddings = tf.nn.l2_normalize(embeddings, axis=-1)

model = Model(inputs=input_layer, outputs=norm_embeddings)


# Create left and right twin models
input1 = Input((64,64,12))
input2 = Input((64,64,12))

left_model = model(input1)
right_model = model(input2)

# Dot product layer
#Change to a classifier or small ann in the future
dot_product = dot([left_model, right_model], axes=1, normalize=False)

#Get the siamese network
siamese_model = Model(inputs=[input1, input2], outputs=dot_product)

prprcs_data = []
for prsn in persons: 
  prprcs_data.append(pre_process("/content/",prsn,order,lowcut,highcut,fs,s,ov))

images = []
for x in prprcs_data:
  try:
    images.append(generate_image_2(x))
  except:
     print("skipped one image")

X1,X2,y = get_pairs(images,pos,neg)

#compile and fit model
siamese_model.compile(optimizer='adam', loss= 'mse', metrics = 'accuracy')
history = siamese_model.fit([np.array(X1), np.array(X2)], np.array(y), epochs=epochs, batch_size=batch_size, shuffle=True, verbose=True, validation_split = validation_split)

#save the models

model.save(os.getcwd()+"/base_model.h5")
siamese_model.save(os.getcwd()+"/full_network.h5")

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("loss_curves.png")

plt.clf()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("accuracy_curves.png")