import scipy.io as io
import os
import numpy as np
import datetime
import pickle
import tensorflow as tf
from keras import layers, losses, Input
from keras.models import Model
from Generator import read_radar_data
from keras2_highway_network import highway_layers

directory = r"S:\Project3_Dataset"
target_directory = directory + '\\Target_Data'
save_dir = 'C:\\Users\\owatkins\\OneDrive - Analog Devices, Inc\\Documents\\Project Folder\\Project 3\\Networks from ' \
          'Python\\'

os.chdir(target_directory)
files = filter(os.path.isfile, os.listdir(target_directory))
files = [os.path.join(target_directory, f) for f in files]
files.sort(key=lambda x: os.path.getmtime(x))

example = np.array(io.loadmat(files[1])["RADAR_Data"])
example_shape = example.shape

num_train_files = np.floor(len(files)*0.6).astype(np.int_)
num_val_files = np.floor(len(files)*0.2).astype(np.int_)
num_test_files = np.floor(len(files)*0.2).astype(np.int_)

train_min = 0
train_max = num_train_files

val_min = train_max + 1
val_max = val_min + num_val_files - 1

test_min = val_max + 1
test_max = len(files) - 1

train_gen = read_radar_data(files[train_min:train_max], example_shape, shuffle=True, batch_size=10)
val_gen = read_radar_data(files[val_min:val_max], example_shape)
test_gen = read_radar_data(files[test_min:test_max], example_shape)

input_tensor = Input(shape=(example_shape[0], example_shape[1], 2))
x = layers.Conv2D(32, 3, activation='relu')(input_tensor)
x = layers.MaxPool2D(pool_size=(3, 3), strides=3)(x)
x = layers.Dense(32, activation='relu')(x)
x = highway_layers(x, 32, activation='relu')
x = highway_layers(x, 32, activation='relu')
x = highway_layers(x, 32, activation='relu')
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPool2D(pool_size=(3, 3), strides=3)(x)
x = layers.Dense(16, activation='relu')(x)
x = highway_layers(x, 16, activation='relu')
x = highway_layers(x, 16, activation='relu')
x = highway_layers(x, 16, activation='relu')
x = layers.Flatten()(x)
output_tensor = layers.Dense(3)(x)

model = Model(input_tensor, output_tensor)

model.compile(loss=losses.mean_absolute_error, optimizer='sgd')
model.summary()

log_dir = save_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit_generator(generator=train_gen, steps_per_epoch=25, epochs=12, validation_data=val_gen,
                              validation_steps=10, verbose=True)
model.save(save_dir + r"\highway_radar_reader.h5")
evaluations = model.evaluate_generator(generator=test_gen, steps=20, verbose=True)
print(history)
print(evaluations)

with open((save_dir + r"\trainHistoryDict"), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

with open((save_dir + r"\trainHistoryDict"), 'wb') as file_pi:
    pickle.dump(evaluations, file_pi)
