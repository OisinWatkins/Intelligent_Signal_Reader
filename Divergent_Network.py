import os
import datetime
import scipy.io as io
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers, losses, Input
from tensorflow.keras.models import Model, Sequential, load_model
import pickle
from Generator import read_radar_data
# os.environ['KERAS_BACKEND'] = 'theano'
# from keras.utils import model_to_dot
# from IPython.display import SVG
# os.environ["PATH"] += os.pathsep + r"C:\Users\owatkins\AppData\Local\Continuum\anaconda3\Lib\site-packages\graphviz" + r"C:\Users\owatkins\AppData\Local\Continuum\anaconda3\Lib\site-packages\pydot"

tf.keras.backend.clear_session()

directory = r"S:\Project3_Dataset"
target_directory = directory + r"\Target_Data"
save_dir = 'C:\\Users\\owatkins\\OneDrive - Analog Devices, Inc\\Documents\\Project Folder\\Project 3\\Networks from ' \
          'Python\\'

os.chdir(target_directory)
files = filter(os.path.isfile, os.listdir(target_directory))
files = [os.path.join(target_directory, f) for f in files]
files.sort(key=lambda x: os.path.getmtime(x))

example = np.array(io.loadmat(files[1])["RADAR_Data"])
example_shape = example.shape

num_train_files = np.floor(len(files) * 0.6).astype(np.int_)
num_val_files = np.floor(len(files) * 0.2).astype(np.int_)
num_test_files = np.floor(len(files) * 0.2).astype(np.int_)

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
preprocess = layers.Conv2D(32, 3, activation='relu')(input_tensor)
preprocess = layers.Conv2D(32, 3, activation='relu')(preprocess)
preprocess = layers.MaxPool2D(pool_size=(3, 3), strides=3)(preprocess)

"""   Range   """
range_m = layers.Conv2D(32, 3, activation='relu')(preprocess)
range_m = layers.Conv2D(32, 3, activation='relu')(range_m)
range_m = layers.MaxPool2D(pool_size=(3, 3), strides=3)(range_m)
range_m = layers.Conv2D(32, 3, activation='relu')(range_m)
range_m = layers.Conv2D(32, 3, activation='relu')(range_m)
range_m = layers.MaxPool2D(pool_size=(3, 3), strides=3)(range_m)
range_m = layers.Conv2D(16, 3, activation='relu')(range_m)
range_m = layers.Conv2D(16, 3, activation='relu')(range_m)
range_m = layers.MaxPool2D(pool_size=(3, 3), strides=3)(range_m)
range_m = layers.Dense(8, activation='relu')(range_m)
range_m = layers.Flatten()(range_m)
range_m = layers.Dense(1)(range_m)

"""   Angle   """
angle_deg = layers.Conv2D(32, 3, activation='relu')(preprocess)
angle_deg = layers.Conv2D(32, 3, activation='relu')(angle_deg)
angle_deg = layers.MaxPool2D(pool_size=(3, 3), strides=3)(angle_deg)
angle_deg = layers.Conv2D(32, 3, activation='relu')(angle_deg)
angle_deg = layers.Conv2D(32, 3, activation='relu')(angle_deg)
angle_deg = layers.MaxPool2D(pool_size=(3, 3), strides=3)(angle_deg)
angle_deg = layers.Conv2D(16, 3, activation='relu')(angle_deg)
angle_deg = layers.Conv2D(16, 3, activation='relu')(angle_deg)
angle_deg = layers.MaxPool2D(pool_size=(3, 3), strides=3)(angle_deg)
angle_deg = layers.Dense(8, activation='relu')(angle_deg)
angle_deg = layers.Flatten()(angle_deg)
angle_deg = layers.Dense(1)(angle_deg)

"""   Velocity   """
velocity_mps = layers.Conv2D(32, 3, activation='relu')(preprocess)
velocity_mps = layers.Conv2D(32, 3, activation='relu')(velocity_mps)
velocity_mps = layers.MaxPool2D(pool_size=(3, 3), strides=3)(velocity_mps)
velocity_mps = layers.Conv2D(32, 3, activation='relu')(velocity_mps)
velocity_mps = layers.Conv2D(32, 3, activation='relu')(velocity_mps)
velocity_mps = layers.MaxPool2D(pool_size=(3, 3), strides=3)(velocity_mps)
velocity_mps = layers.Conv2D(16, 3, activation='relu')(velocity_mps)
velocity_mps = layers.Conv2D(16, 3, activation='relu')(velocity_mps)
velocity_mps = layers.MaxPool2D(pool_size=(3, 3), strides=3)(velocity_mps)
velocity_mps = layers.Dense(8, activation='relu')(velocity_mps)
velocity_mps = layers.Flatten()(velocity_mps)
velocity_mps = layers.Dense(1)(velocity_mps)

concat_out = layers.Concatenate(axis=-1)([range_m, angle_deg, velocity_mps])
model = Model(input_tensor, concat_out)

# trunk = Sequential()
# trunk.add(layers.Conv2D(32, 3, activation='relu', input_shape=(example_shape[0], example_shape[1], 2)))
#
# range_m = Sequential()
# range_m.add(layers.Conv2D(32, 3, activation='relu'))
# range_m.add(layers.Conv2D(32, 3, activation='relu'))
# range_m.add(layers.MaxPool2D(pool_size=(3, 3), strides=3))
# range_m.add(layers.Conv2D(32, 3, activation='relu'))
# range_m.add(layers.Conv2D(32, 3, activation='relu'))
# range_m.add(layers.MaxPool2D(pool_size=(3, 3), strides=3))
# range_m.add(layers.Conv2D(16, 3, activation='relu'))
# range_m.add(layers.Conv2D(16, 3, activation='relu'))
# range_m.add(layers.MaxPool2D(pool_size=(3, 3), strides=3))
# range_m.add(layers.Dense(8, activation='relu'))
# head1 = Sequential([trunk, range_m])
#
# angle_deg = Sequential()
# angle_deg.add(layers.Conv2D(32, 3, activation='relu'))
# angle_deg.add(layers.Conv2D(32, 3, activation='relu'))
# angle_deg.add(layers.MaxPool2D(pool_size=(3, 3), strides=3))
# angle_deg.add(layers.Conv2D(32, 3, activation='relu'))
# angle_deg.add(layers.Conv2D(32, 3, activation='relu'))
# angle_deg.add(layers.MaxPool2D(pool_size=(3, 3), strides=3))
# angle_deg.add(layers.Conv2D(16, 3, activation='relu'))
# angle_deg.add(layers.Conv2D(16, 3, activation='relu'))
# angle_deg.add(layers.MaxPool2D(pool_size=(3, 3), strides=3))
# angle_deg.add(layers.Dense(8, activation='relu'))
# head2 = Sequential([trunk, angle_deg])
#
# velocity_mps = Sequential()
# velocity_mps.add(layers.Conv2D(32, 3, activation='relu'))
# velocity_mps.add(layers.Conv2D(32, 3, activation='relu'))
# velocity_mps.add(layers.MaxPool2D(pool_size=(3, 3), strides=3))
# velocity_mps.add(layers.Conv2D(32, 3, activation='relu'))
# velocity_mps.add(layers.Conv2D(32, 3, activation='relu'))
# velocity_mps.add(layers.MaxPool2D(pool_size=(3, 3), strides=3))
# velocity_mps.add(layers.Conv2D(16, 3, activation='relu'))
# velocity_mps.add(layers.Conv2D(16, 3, activation='relu'))
# velocity_mps.add(layers.MaxPool2D(pool_size=(3, 3), strides=3))
# velocity_mps.add(layers.Dense(8, activation='relu'))
# head3 = Sequential([trunk, velocity_mps])
#
# model = Sequential()
# model.add(layers.Concatenate([head1, head2, head3]))
# model.add(layers.Flatten())
# model.add(layers.Dense(3))

model.compile(loss=losses.mean_absolute_error, optimizer='sgd', metrics=['accuracy'])
model.summary()
# SVG(model_to_dot(model).create(prog='dot', format='svg'))
plot_model(model, to_file=directory + "model_plot.png", show_shapes=True, show_layer_names=True)

# model.save(directory + '\\UNTRAINED_divergent_radar_reader.h5')
log_dir = save_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(x=train_gen, steps_per_epoch=25, epochs=12, validation_data=val_gen, validation_steps=10,
                    verbose=True, callbacks=[tensorboard_callback])

model.save(save_dir + r"\divergent_radar_reader.h5")

evaluations = model.evaluate_generator(generator=test_gen, steps=20, verbose=True)
print(history)
print(evaluations)

with open((save_dir + '\\trainHistoryDict'), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

with open((save_dir + '\\trainHistoryDict'), 'wb') as file_pi:
    pickle.dump(evaluations, file_pi)

os.system(r'tensorboard --logdir r"C:\Users\owatkins\OneDrive - Analog Devices, Inc\Documents\Project Folder\Project '
          r'3\Networks from Python\"')
