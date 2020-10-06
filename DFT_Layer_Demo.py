"""

"""
import os
import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import warnings
import numpy as np
import tensorflow as tf
from scipy.io import wavfile
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras import constraints
from tensorflow.python.keras import regularizers
from Fourier_Transform import DFT
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    print("\n\t---------    Running Demo for the DFT Layer    ---------\n")

    print("\n>>\n"
          ">> This file will run a bespoke model to handle the following task:\n"
          ">> -> `Speach-to-Text conversion`\n"
          ">>\n"
          ">> To accomplish this, I will define 2 models:\n"
          ">> -> One with the DFT Layer high up in the architecture.\n"
          ">> -> One using more typical Machine Learning Practices.\n"
          ">>\n"
          ">> After training has completed I will run each model through an extensive test\n"
          ">> to determine whether or not the DFT layer bears any benefit to signal processing networks.\n"
          ">> The typical Machine Learning model will be pulled from blogposts on the internet. Doing this\n"
          ">> should ensure I'm comparing my work to the tried and tested models used in the world today.\n"
          ">>\n"
          ">> The data I'm using for this application comes from Kaggle:\n"
          ">> https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data"
          ">>\n")

    print("\n>>\n"
          ">> First, we’ll visualize the audio signal in the time domain:\n"
          ">>\n")
    train_audio_path = 'C:\\Users\\owatkins\\OneDrive - Analog Devices, Inc\\Documents\\Project Folder\\Project 3\\' \
                       'tensorflow-speech-recognition-challenge\\train\\audio'
    samples, sample_rate = librosa.load(train_audio_path + 'yes/0a7c2a8d_nohash_0.wav', sr=16000)
    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(211)
    ax1.set_title('Raw wave of ' + '../train/audio/yes/0a7c2a8d_nohash_0.wav')
    ax1.set_xlabel('time')
    ax1.set_ylabel('Amplitude')
    ax1.plot(np.linspace(0, sample_rate / len(samples), sample_rate), samples)

    # Let us now look at the sampling rate of the audio signals
    ipd.Audio(samples, rate=sample_rate)
    print(sample_rate)

    # From the above, we can understand that the sampling rate of the signal is 16000 Hz. Let us resample it to 8000 Hz
    # as typically human speech is sampled at 8kHz
    samples = librosa.resample(samples, sample_rate, 8000)
    ipd.Audio(samples, rate=8000)

    print("\n>>\n"
          ">> Now, let’s understand the number of recordings for each voice command:\n"
          ">>\n")
    labels = os.listdir(train_audio_path)
    # find count of each label and plot bar graph
    no_of_recordings = []
    for label in labels:
        waves = [f for f in os.listdir(train_audio_path + '/' + label) if f.endswith('.wav')]
        no_of_recordings.append(len(waves))

    # plot
    plt.figure(figsize=(30, 5))
    index = np.arange(len(labels))
    plt.bar(index, no_of_recordings)
    plt.xlabel('Commands', fontsize=12)
    plt.ylabel('No of recordings', fontsize=12)
    plt.xticks(index, labels, fontsize=15, rotation=60)
    plt.title('No. of recordings for each command')
    plt.show()

    labels = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

    print("\n>>\n"
          ">> What’s next? A look at the distribution of the duration of recordings:\n"
          ">>\n")
    duration_of_recordings = []
    for label in labels:
        waves = [f for f in os.listdir(train_audio_path + '/' + label) if f.endswith('.wav')]
        for wav in waves:
            sample_rate, samples = wavfile.read(train_audio_path + '/' + label + '/' + wav)
            duration_of_recordings.append(float(len(samples) / sample_rate))

    plt.hist(np.array(duration_of_recordings))

    print("\n>>\n"
          ">> In the data exploration part earlier, we have seen that the duration of a few recordings is less than 1\n"
          ">> second and the sampling rate is too high. So, let us read the audio waves and use the below-preprocessing\n"
          ">> steps to deal with this. Here are the two steps we’ll follow:\n"
          ">>\n"
          ">> -> Resampling\n"
          ">> -> Removing shorter commands of less than 1 second\n"
          ">>\n"
          ">> Let us define these preprocessing steps in the below code snippet:\n"
          ">>\n")

    all_wave = []
    all_label = []
    for label in labels:
        print(label)
        waves = [f for f in os.listdir(train_audio_path + '\\' + label) if f.endswith('.wav')]
        for wav in waves:
            samples, sample_rate = librosa.load(train_audio_path + '\\' + label + '\\' + wav, sr=16000)
            samples = librosa.resample(samples, sample_rate, 8000)
            if (len(samples) == 8000):
                all_wave.append(samples)
                all_label.append(label)

    print("\n>>\n"
          ">> Convert the output labels to integer encoded:\n"
          ">>\n")
    le = LabelEncoder()
    y = le.fit_transform(all_label)
    classes = list(le.classes_)

    y = tf.keras.utils.to_categorical(y, num_classes=len(labels))

    # Reshape the 2D array to 3D since the input to the conv1d must be a 3D array:
    all_wave = np.array(all_wave).reshape(-1, 8000, 1)

    print("\n>>\n"
          ">> Next, we will train the model on 80% of the data and validate on the remaining 20%:\n"
          ">>\n")
    x_tr, x_val, y_tr, y_val = train_test_split(np.array(all_wave), np.array(y), stratify=y, test_size=0.2,
                                                random_state=777, shuffle=True)

    # Define The standard Model, no DFT
    inputs = Input(shape=(8000, 1))

    # First Conv1D layer
    conv = layers.Conv1D(8, 13, padding='valid', activation='relu', strides=1)(inputs)
    conv = layers.MaxPooling1D(3)(conv)
    conv = layers.Dropout(0.3)(conv)

    # Second Conv1D layer
    conv = layers.Conv1D(16, 11, padding='valid', activation='relu', strides=1)(conv)
    conv = layers.MaxPooling1D(3)(conv)
    conv = layers.Dropout(0.3)(conv)

    # Third Conv1D layer
    conv = layers.Conv1D(32, 9, padding='valid', activation='relu', strides=1)(conv)
    conv = layers.MaxPooling1D(3)(conv)
    conv = layers.Dropout(0.3)(conv)

    # Fourth Conv1D layer
    conv = layers.Conv1D(64, 7, padding='valid', activation='relu', strides=1)(conv)
    conv = layers.MaxPooling1D(3)(conv)
    conv = layers.Dropout(0.3)(conv)

    # Flatten layer
    conv = layers.Flatten()(conv)

    # Dense Layer 1
    conv = layers.Dense(256, activation='relu')(conv)
    conv = layers.Dropout(0.3)(conv)

    # Dense Layer 2
    conv = layers.Dense(128, activation='relu')(conv)
    conv = layers.Dropout(0.3)(conv)

    outputs = layers.Dense(len(labels), activation='softmax')(conv)

    model = Model(inputs, outputs)
    model.summary()

    # Define the loss function to be categorical cross-entropy since it is a multi-classification problem:
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Early stopping and model checkpoints are the callbacks to stop training the neural network at the right time and
    # to save the best model after every epoch:
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001)
    mc = ModelCheckpoint("C:\\Users\\owatkins\\OneDrive - Analog Devices, Inc\\Documents\\Project Folder\\Project 3\\Code\\"
               "Intelligent_Signal_Reader\\best_model.hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    history = model.fit(x_tr, y_tr, epochs=100, callbacks=[es, mc], batch_size=32, validation_data=(x_val, y_val))

    model.save("C:\\Users\\owatkins\\OneDrive - Analog Devices, Inc\\Documents\\Project Folder\\Project 3\\Code\\"
               "Intelligent_Signal_Reader\\standard_model.h5")

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
