import scipy.io as io
import os
# os.environ['KERAS_BACKEND'] = 'theano'
import numpy as np
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Sequential, load_model
from Generator import read_radar_data

directory = 'C:\\Users\\owatkins\\OneDrive - Analog Devices, Inc\\Documents\\Project Folder\\Project 3\\Dataset'
target_directory = directory + '\\Target_Data'

os.chdir(target_directory)
files = filter(os.path.isfile, os.listdir(target_directory))
files = [os.path.join(target_directory, f) for f in files]
files.sort(key=lambda x: os.path.getmtime(x))

example = np.array(io.loadmat(files[1])["RADAR_Data"])
example_shape = example.shape

num_train_files = int(np.floor(len(files)*0.6))
num_val_files = int(np.floor(len(files)*0.2))
num_test_files = int(np.floor(len(files)*0.2))

train_min = int(0)
train_max = int(num_train_files)

val_min = int(train_max + 1)
val_max = int(val_min + num_val_files)

test_min = int(val_max + 1)
test_max = int(len(files) - 1)

train_gen = read_radar_data(files[train_min:train_max], example_shape, shuffle=True, batch_size=10)
val_gen = read_radar_data(files[val_min:val_max], example_shape)
test_gen = read_radar_data(files[test_min:test_max], example_shape)

model = Sequential()
model.add(layers.Conv2D(32, 2, activation='relu', input_shape=(example_shape[0], example_shape[1], 2)))
model.add(layers.MaxPool2D(pool_size=(3, 3), strides=3))
model.add(layers.Conv2D(32, 2, activation='relu'))
model.add(layers.MaxPool2D(pool_size=(3, 3), strides=3))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.MaxPool2D(pool_size=(3, 3), strides=3))
model.add(layers.Dropout(0.25))
model.add(layers.Flatten())
model.add(layers.Dense(3))

model.compile(loss=losses.mean_absolute_error, optimizer='sgd')
model.summary()

history = model.fit(x=train_gen, steps_per_epoch=25, epochs=12, validation_data=val_gen, validation_steps=10,
                    verbose=True)
model.save(directory + '\\conv2d_radar_reader.h5')
evaluations = model.evaluate_generator(generator=test_gen, steps=20, verbose=True)
print(evaluations)

"""
model = load_model(directory + '\\conv2d_radar_reader.h5')
mean_absolute_percentage_error = np.zeros(100, 2)
for i, presentation in enumerate(test_gen):
    if i >= 100:
        break
    batch = i*5
    range_absolute_error = np.absolute(np.subtract(presentation[1][:, 0], predictions[batch:batch+4, 0]))
    range_absolute_percentage_error = np.nan_to_num(np.multiply(np.divide(range_absolute_error, presentation[1][:, 0]), 100))
    range_mean_absolute_percentage_error = np.mean(range_absolute_percentage_error)
    velocity_absolute_error = np.absolute(np.subtract(presentation[1][:, 2], predictions[batch:batch + 5, 0]))
    velocity_absolute_percentage_error = np.nan_to_num(np.multiply(np.divide(velocity_absolute_error, presentation[1][:, 2]), 100))
    velocity_mean_absolute_percentage_error = np.mean(velocity_absolute_percentage_error)

    mean_absolute_percentage_error[batch:batch+4, 0] = range_mean_absolute_percentage_error
    mean_absolute_percentage_error[batch:batch + 4, 1] = velocity_mean_absolute_percentage_error
"""
