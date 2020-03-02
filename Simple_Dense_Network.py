import scipy.io as io
import os
import numpy as np
from keras import layers
from keras import losses
from keras.models import Sequential
from Generator import read_radar_data

directory = 'C:\\Dataset'
target_directory = directory + '\\Target_Data'

os.chdir(target_directory)
files = filter(os.path.isfile, os.listdir(target_directory))
files = [os.path.join(target_directory, f) for f in files]
files.sort(key=lambda x: os.path.getmtime(x))

example = np.array(io.loadmat(files[1])["RADAR_Data"])
example_shape = example.shape

num_train_files = np.floor(len(files)*0.6)
num_val_files = np.floor(len(files)*0.2)
num_test_files = np.floor(len(files)*0.2)

train_min = 0
train_max = num_train_files

val_min = train_max + 1
val_max = val_min + num_val_files

test_min = val_max + 1
test_max = len(files) - 1

train_gen = read_radar_data(files[train_min:train_max], example_shape, shuffle=True, batch_size=10)
val_gen = read_radar_data(files[val_min:val_max], example_shape)
test_gen = read_radar_data(files[test_min:test_max], example_shape)

model = Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(example_shape[0], example_shape[1], 2)))
model.add(layers.Flatten())
model.add(layers.Dense(3))

model.summary()
model.compile(loss=losses.mean_squared_error, optimizer='sgd')

history = model.fit_generator(generator=train_gen, steps_per_epoch=50, epochs=12, validation_data=val_gen,
                              validation_steps=num_val_files, verbose=True)
model.save(directory + '\\dense_radar_reader.h5')
predictions = model.predict_generator(generator=test_gen, verbose=True)
