"""

"""
import os
import time
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
from Fourier_Transform import DFT, Wnp
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    print("\n\t---------    Running Demo for the DFT Layer    ---------\n")

    print("\n>>\n"
          ">> This file will run a bespoke model to handle the following task:\n"
          ">> -> `Speech Recognition`\n"
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
          ">> https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data\n"
          ">>\n"
          ">> The Standard model I'm using comes from:\n"
          ">> https://github.com/aravindpai/Speech-Recognition/blob/master/Speech%20Recognition.ipynb\n"
          ">>\n")

    print("\n>>\n"
          ">> First, we’ll visualize the audio signal in the time domain:\n"
          ">>\n")
    train_audio_path = 'C:\\Users\\owatkins\\OneDrive - Analog Devices, Inc\\Documents\\Project Folder\\Project 3\\' \
                       'tensorflow-speech-recognition-challenge\\train\\audio'
    samples, sample_rate = librosa.load(train_audio_path + '\\yes\\0a7c2a8d_nohash_0.wav', sr=16000)
    # fig = plt.figure(figsize=(14, 8))
    # ax1 = fig.add_subplot(211)
    # ax1.set_title('Raw wave of ' + '../train/audio/yes/0a7c2a8d_nohash_0.wav')
    # ax1.set_xlabel('time')
    # ax1.set_ylabel('Amplitude')
    # ax1.plot(np.linspace(0, sample_rate / len(samples), sample_rate), samples)

    # Let us now look at the sampling rate of the audio signals
    ipd.Audio(samples, rate=sample_rate)
    print(f"Audio Sampling rate: {sample_rate} Hz")

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
        waves = [f for f in os.listdir(train_audio_path + '\\' + label) if f.endswith('.wav')]
        no_of_recordings.append(len(waves))

    # plot
    # plt.figure(figsize=(30, 5))
    # index = np.arange(len(labels))
    # plt.bar(index, no_of_recordings)
    # plt.xlabel('Commands', fontsize=12)
    # plt.ylabel('No of recordings', fontsize=12)
    # plt.xticks(index, labels, fontsize=15, rotation=60)
    # plt.title('No. of recordings for each command')
    # plt.show()

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

    print("\n>>\n"
          ">> In the data exploration part earlier, we have seen that the duration of a few recordings is less than 1\n"
          ">> second and the sampling rate is too high. So, let us read the audio waves and use the below-preprocessing\n"
          ">> steps to deal with this. Here are the two steps we’ll follow:\n"
          ">>\n"
          ">> -> Resampling\n"
          ">> -> Removing commands shorter than 1 second\n"
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
    all_wave = np.array(all_wave)  # .reshape(-1, 8000, 1)
    print(f"Training Data Shape: {all_wave.shape}")
    print(f"Presentation Shape: {all_wave[0].shape}")

    print("\n>>\n"
          ">> Next, we will train the model on 80% of the data and validate on the remaining 20%:\n"
          ">>\n")
    x_tr, x_val, y_tr, y_val = train_test_split(all_wave, np.array(y), stratify=y, test_size=0.2,
                                                random_state=777, shuffle=True)
                                                
    def training_generator(batch_size, inputs, outputs, augment=True):
    
        max_index = np.floor(all_wave.shape[0] * 0.8)
    
        def add_noise(data, noise_factor):
            noise = np.random.randn(len(data))
            augmented_data = data + noise_factor * noise
            # Cast back to same data type
            augmented_data = augmented_data.astype(type(data[0]))
            return augmented_data
            
        def time_shift(data, sampling_rate, shift_max, shift_direction):
            shift = np.random.randint(sampling_rate * shift_max)
            if shift_direction == 'right':
                shift = -shift
            elif shift_direction == 'both':
                direction = np.random.randint(0, 2)
                if direction == 1:
                    shift = -shift
            augmented_data = np.roll(data, shift)
            # Set to silence for heading/ tailing
            if shift > 0:
                augmented_data[:shift] = 0
            else:
                augmented_data[shift:] = 0
            return augmented_data
            
        def pitch_shift(data, sampling_rate, pitch_factor):
            return librosa.effects.pitch_shift(data[0, :], sampling_rate, pitch_factor)
            
        # def speed_change(data, speed_factor):
            # return librosa.effects.time_stretch(data, speed_factor)
            
        while True:
            batch_samples = []
            batch_targets = []
            
            for i in range(batch_size):
                index = np.random.randint(0, high=max_index)
                
                noise_f = np.random.rand(1, 1)
                time_shift_max = 0.3
                pitch_s = np.random.rand(1, 1)
                speed_c = 2 * np.random.rand(1, 1)
                
                augmented_input = add_noise(inputs[index], noise_f)
                augmented_input = time_shift(augmented_input, 8000, time_shift_max, 'both')
                augmented_input = pitch_shift(augmented_input, 8000, pitch_s)
                # augmented_input = speed_change(augmented_input, speed_c)
                
                batch_samples.append(augmented_input)
                batch_targets.append(outputs[index])
                
            yield np.array(batch_samples), batch_targets
            
    # train_gen = training_generator(32, x_tr, y_tr)
    
    # print("Training generator defined and instantiated\n")

    start_time = time.time()
    # Define The standard Model, no DFT

    # inputs = Input(shape=(8000, 1))

    # # Standard Model
    # # First Conv1D layer
    # conv = layers.Conv1D(8, 13, padding='valid', activation='relu', strides=1)(inputs)
    # conv = layers.MaxPooling1D(3)(conv)
    # conv = layers.Dropout(0.3)(conv)
    
    # # Second Conv1D layer
    # conv = layers.Conv1D(16, 11, padding='valid', activation='relu', strides=1)(conv)
    # conv = layers.MaxPooling1D(3)(conv)
    # conv = layers.Dropout(0.3)(conv)
    
    # # Third Conv1D layer
    # conv = layers.Conv1D(32, 9, padding='valid', activation='relu', strides=1)(conv)
    # conv = layers.MaxPooling1D(3)(conv)
    # conv = layers.Dropout(0.3)(conv)
    
    # # Fourth Conv1D layer
    # conv = layers.Conv1D(64, 7, padding='valid', activation='relu', strides=1)(conv)
    # conv = layers.MaxPooling1D(3)(conv)
    # conv = layers.Dropout(0.3)(conv)
    
    # # Flatten layer
    # conv = layers.Flatten()(conv)
    
    # # Dense Layer 1
    # conv = layers.Dense(256, activation='relu')(conv)
    # conv = layers.Dropout(0.3)(conv)
    
    # # Dense Layer 2
    # conv = layers.Dense(128, activation='relu')(conv)
    # conv = layers.Dropout(0.3)(conv)

    # outputs = layers.Dense(len(labels), activation='softmax')(conv)
    
    # model = Model(inputs, outputs)
    # model.summary()
    
    # # Define the loss function to be categorical cross-entropy since it is a multi-classification problem:
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Define The DFT Model
    inputs = Input(shape=(8000, ))
    padding = tf.constant([[0, 0], [32, 32]])
    sig_t = tf.pad(inputs, padding, 'CONSTANT')

    sig_t_split = tf.split(sig_t, num_or_size_splits=63, axis=1)
    
    twiddle_init = []
    for i in range(128):
        row = []
        for j in range(128):
            row.append(Wnp(N=128, p=(i * j)))
        twiddle_init.append(row)
        
    twiddle_init = np.array(twiddle_init)

    sig_freq_0 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[0])
    sig_freq_1 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[1])
    sig_freq_2 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[2])
    sig_freq_3 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[3])
    sig_freq_4 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[4])
    sig_freq_5 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[5])
    sig_freq_6 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[6])
    sig_freq_7 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[7])
    sig_freq_8 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[8])
    sig_freq_9 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[9])
    sig_freq_10 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[10])
    print("First 10 DFT's done...")

    sig_freq_11 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[11])
    sig_freq_12 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[12])
    sig_freq_13 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[13])
    sig_freq_14 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[14])
    sig_freq_15 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[15])
    sig_freq_16 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[16])
    sig_freq_17 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[17])
    sig_freq_18 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[18])
    sig_freq_19 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[19])
    sig_freq_20 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[20])
    print("First 20 DFT's done...")

    sig_freq_21 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[21])
    sig_freq_22 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[22])
    sig_freq_23 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[23])
    sig_freq_24 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[24])
    sig_freq_25 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[25])
    sig_freq_26 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[26])
    sig_freq_27 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[27])
    sig_freq_28 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[28])
    sig_freq_29 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[29])
    sig_freq_30 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[30])
    print("First 30 DFT's done...")

    sig_freq_31 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[31])
    sig_freq_32 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[32])
    sig_freq_33 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[33])
    sig_freq_34 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[34])
    sig_freq_35 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[35])
    sig_freq_36 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[36])
    sig_freq_37 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[37])
    sig_freq_38 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[38])
    sig_freq_39 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[39])
    sig_freq_40 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[40])
    print("First 40 DFT's done...")

    sig_freq_41 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[41])
    sig_freq_42 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[42])
    sig_freq_43 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[43])
    sig_freq_44 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[44])
    sig_freq_45 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[45])
    sig_freq_46 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[46])
    sig_freq_47 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[47])
    sig_freq_48 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[48])
    sig_freq_49 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[49])
    sig_freq_50 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[50])
    print("First 50 DFT's done...")

    sig_freq_51 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[51])
    sig_freq_52 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[52])
    sig_freq_53 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[53])
    sig_freq_54 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[54])
    sig_freq_55 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[55])
    sig_freq_56 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[56])
    sig_freq_57 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[57])
    sig_freq_58 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[58])
    sig_freq_59 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[59])
    sig_freq_60 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[60])
    sig_freq_61 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[61])
    sig_freq_62 = DFT(num_samples=128, twiddle_initialiser=twiddle_init.copy())(sig_t_split[62])
    print("All DFT's done!")

    sig_freq_abs = tf.abs(tf.stack([sig_freq_0, sig_freq_1, sig_freq_2, sig_freq_3, sig_freq_4, sig_freq_5, sig_freq_6,
                                    sig_freq_7, sig_freq_8, sig_freq_9, sig_freq_1, sig_freq_1, sig_freq_1, sig_freq_13,
                                    sig_freq_14, sig_freq_15, sig_freq_16, sig_freq_17, sig_freq_18, sig_freq_19, sig_freq_20,
                                    sig_freq_21, sig_freq_22, sig_freq_23, sig_freq_24, sig_freq_25, sig_freq_26, sig_freq_27,
                                    sig_freq_28, sig_freq_29, sig_freq_30, sig_freq_31, sig_freq_32, sig_freq_33, sig_freq_34,
                                    sig_freq_35, sig_freq_36, sig_freq_37, sig_freq_38, sig_freq_39, sig_freq_40, sig_freq_41,
                                    sig_freq_42, sig_freq_43, sig_freq_44, sig_freq_45, sig_freq_46, sig_freq_47, sig_freq_48,
                                    sig_freq_49, sig_freq_50, sig_freq_51, sig_freq_52, sig_freq_53, sig_freq_54, sig_freq_55,
                                    sig_freq_56, sig_freq_57, sig_freq_58, sig_freq_59, sig_freq_60, sig_freq_61, sig_freq_62]))

    sig_freq_abs_transpose = tf.transpose(sig_freq_abs, perm=(1, 0, 2))
    print("DFT Stack Complete")

    dropout0 = layers.Dropout(0.5)(sig_freq_abs_transpose)
    norm0 = layers.BatchNormalization(axis=1)(dropout0)
    
    conv1 = layers.SeparableConv1D(512, kernel_size=(4), activation='relu')(norm0)
    maxpool1 = layers.MaxPooling1D(4)(conv1)
    dropout1 = layers.Dropout(0.3)(maxpool1)
    norm1 = layers.BatchNormalization(axis=1)(dropout1)

    conv2 = layers.SeparableConv1D(512, kernel_size=(4), activation='relu')(norm1)
    maxpool2 = layers.MaxPooling1D(2)(conv2)
    dropout2 = layers.Dropout(0.3)(maxpool2)
    norm2 = layers.BatchNormalization(axis=1)(dropout2)
    
    conv3 = layers.SeparableConv1D(256, kernel_size=(4), activation='relu')(norm2)
    maxpool3 = layers.MaxPooling1D(2)(conv3)
    dropout3 = layers.Dropout(0.3)(maxpool3)
    norm3 = layers.BatchNormalization(axis=1)(dropout3)
    
    flatten = layers.Flatten()(norm3)
    dense0 = layers.Dense(128, activation='relu')(flatten)
    dropout4 = layers.Dropout(0.3)(dense0)
    
    dense1 = layers.Dense(64, activation='relu')(dropout4)
    dropout5 = layers.Dropout(0.3)(dense1)
    
    dense2 = layers.Dense(32, activation='relu')(dropout5)
    dropout6 = layers.Dropout(0.3)(dense2)
    
    outputs = layers.Dense(len(labels), activation='softmax')(dropout6)

    model = Model(inputs, outputs)

    print(f"\n\t--------- Model building took {(time.time() - start_time)} seconds ---------\n")

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Early stopping and model checkpoints are the callbacks to stop training the neural network at the right time and
    # to save the best model after every epoch:
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5, min_delta=0.0001)
    mc = ModelCheckpoint("C:\\Users\\owatkins\\OneDrive - Analog Devices, Inc\\Documents\\Project Folder\\Project 3\\Code\\"
               "Intelligent_Signal_Reader\\best_model.hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    history = model.fit(x_tr, y_tr, epochs=100, callbacks=[es], batch_size=32, validation_data=(x_val, y_val))
    # history = model.fit(train_gen, epochs=100, callbacks=[es], steps_per_epoch=50, validation_data=(x_val, y_val))

    model.save("C:\\Users\\owatkins\\OneDrive - Analog Devices, Inc\\Documents\\Project Folder\\Project 3\\Code\\"
               "Intelligent_Signal_Reader\\DFT_model.h5", include_optimizer=False)
               
    print("Model Saved!")

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.legend()
    plt.show()
