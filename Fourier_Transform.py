"""
Author: Oisín Watkins
Email: oisinwatkins97@gmail.com
"""

import os
import csv
import math
import numpy as np
import tensorflow as tf
from scipy.fftpack import fft
from matplotlib import rc, pyplot as plt
from tensorflow.keras import layers, Model, losses, Input
# tf.config.experimental_run_functions_eagerly(True)


def generate_twiddle_arrays(directory: str = os.getcwd(), lowest_power_of_2: int = 1, highest_power_of_2: int = 1):
    """
    Function generates csv files to contain the twiddle arrays for later use, makes creating objects of DFT layer faster
    generate_twiddle_arrays(directory='C:\\Users\\owatkins\\OneDrive - Analog Devices, Inc\\Documents\\Project Folder\\
    Project 3\\Code\\Python\\Default Twiddle Arrays', highest_power_of_2=3)

    :param directory: Folder to store twiddle array csv's in
    :param lowest_power_of_2: The lowest power of 2 needed for generation.
    :param highest_power_of_2: The highest power of 2 needed for generation.
    :return: None
    """
    print(tf.executing_eagerly())
    print(f'Saving Twiddle Arrays to: {directory}...')
    for power in range(lowest_power_of_2, highest_power_of_2 + 1):
        num_samples = 2 ** power
        W = []
        for i in range(num_samples):
            row = []
            for j in range(num_samples):
                row.append(Wnp(N=num_samples, p=(i * j)).numpy())
            W.append(row)
        file_dir = directory + '\\' + 'TA_2^' + str(power) + '.csv'
        print(f'Now Saving: {file_dir}')
        # tf.io.write_file(filename=file_dir, contents=W)
        with open(file_dir, 'w+') as f:
            csvWriter = csv.writer(f, delimiter=',')
            csvWriter.writerows(W)
    print('Saving Complete')


def plot_signal(axs, time_stamp, sig_freq, NFFT, clean_sig=None, noisy_sig=None, clean_fft=None, noisy_fft=None):
    """
    This function plots a clean and a noisy signla with the respective ffts in a single window for comparison.

    :param axs: Handle for the 4 axes used to plot the information
    :param time_stamp: Array holding the time stamps for each time domain signal sample
    :param sig_freq: Real value of the frequency of the signal of interest.
    :param NFFT: The length of the DFT/FFT
    :param clean_sig: Array holding the clean signal without noise.
    :param noisy_sig: Array holding the signal distorted with noise
    :param clean_fft: Array holding the fft of the clean signal
    :param noisy_fft: Array holding the fft of the noisy signal
    :return: None
    """
    if clean_sig is not None:
        axs[0, 0].plot(time_stamp, clean_sig)  # plot using pyplot library from matplotlib package
        axs[0, 0].set_title(f'Clean Sine wave f = {sig_freq} Hz')  # plot title
        axs[0, 0].set_xlabel('Time (s)')  # x-axis label
        axs[0, 0].set_ylabel('Amplitude')  # y-axis label

    if clean_fft is not None:
        nVals = np.arange(start=0, stop=NFFT)  # raw index for FFT plot
        axs[0, 1].plot(nVals, np.abs(clean_fft))
        axs[0, 1].set_title('Clean_Sig Double Sided FFT - without FFTShift')
        axs[0, 1].set_xlabel('Sample points (N-point DFT)')
        axs[0, 1].set_ylabel('DFT Values')

    if noisy_sig is not None:
        axs[1, 0].plot(time_stamp, noisy_sig)  # plot using pyplot library from matplotlib package
        axs[1, 0].set_title('Noisy Sine wave f=' + str(sig_freq) + ' Hz')  # plot title
        axs[1, 0].set_xlabel('Time (s)')  # x-axis label
        axs[1, 0].set_ylabel('Amplitude')  # y-axis label

    if noisy_fft is not None:
        nVals = np.arange(start=0, stop=NFFT)  # raw index for FFT plot
        axs[1, 1].plot(nVals, np.abs(noisy_fft))
        axs[1, 1].set_title('Noisy_Sig Double Sided FFT - without FFTShift')
        axs[1, 1].set_xlabel('Sample points (N-point DFT)')
        axs[1, 1].set_ylabel('DFT Values')


def sine_wave(f, overSampRate, phase, nCyl):
    """
    Generate sine wave signal.
    Example:
    f=10; overSampRate=30;
    phase = 1/3*np.pi;nCyl = 5;
    (t,g) = sine_wave(f,overSampRate,phase,nCyl)

    :param f : frequency of sine wave in Hertz
    :param overSampRate : oversampling rate (integer)
    :param phase : desired phase shift in radians
    :param nCyl : number of cycles of sine wave to generate
    :return (t,g) : time base (t) and the signal g(t) as tuple
    """
    fs = overSampRate*f  # sampling frequency
    t = np.arange(0, nCyl*1/f-1/fs, 1/fs)  # time base
    g = np.sin(2*np.pi*f*t+phase)  # replace with cos if a cosine wave is desired
    return t, g  # return time base and signal g(t) as tuple


def next_power_of_2(x):
    """
    Given any input value, x, next_power_of_2 returns the next power of 2 above the input

    :param x: Input integer
    :return: The next integer power of 2 above the input value (if x is already an integer power of 2 return x)
    """
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def Wnp(N, p):
    """
    Function makes one Twiddle Factor needed for the Fourier Transform algorithm

    :param N: Length of the Fourier Transform input sequence
    :param p: root number for this particular twiddle factor
    :return: twiddle factor: e ^ -j * ((2 * pi * p) / N)
    """
    return tf.math.exp(tf.multiply(tf.complex(0.0, -1.0), tf.complex((2 * np.pi * p / N), 0.0)))


@tf.custom_gradient
def fft_custom(inputs, tuning_radii=None, tuning_angles=None):
    """
    Performs FFT algorithm recursively on the inputs using the tuning_radii and tuning_angles.

    :param inputs: Input Signal (Real or Complex data acceptable)
    :param tuning_radii: Tensor from the Layer object, used to tune the magnitude of each twiddle factor
    :param tuning_angles: Tensor from the Layer object, used to tune the angle of each twiddle factor
    :return: FFT of the Input Signal (dtype = tf.float32)
    """

    if not tf.is_tensor(inputs):
        inputs = tf.convert_to_tensor(inputs, dtype=tf.complex64)

    if not (inputs.dtype == tf.complex64):
        inputs = tf.cast(inputs, tf.complex64)

    N = inputs.shape.as_list()[0]
    if N is not None:
        if N <= 1:
            def grad():
                return 0, 0, 0
            return inputs, grad
        if not math.log2(N).is_integer():
            num_zeros_to_add = next_power_of_2(N) - N
            inputs = tf.concat([inputs, tf.zeros(num_zeros_to_add, dtype=tf.complex64)])
            N = inputs.shape.as_list()[0]

        even = fft(inputs[0::2])
        odd = fft(inputs[1::2])

        T = [tf.math.multiply(Wnp(N, k), tf.cast(odd[k], dtype=tf.complex64)) for k in range(N // 2)]

        def grad():
            return 0, 1, 1

        return tf.abs([tf.add(tf.cast(even[k], dtype=tf.complex64), T[k]) for k in range(N // 2)] + [tf.subtract(tf.cast(even[k], dtype=tf.complex64), T[k]) for k in range(N // 2)], name='fft_calc'), grad
    else:
        def grad():
            return 0, 0, 0
        return tf.convert_to_tensor([0], dtype=tf.complex64), grad


@tf.custom_gradient
def dft(inputs: tf.Tensor or list or np.ndarray, twiddle_array: tf.Tensor or list or np.ndarray = None,
        verbose: bool = False, return_real: bool = True):
    """
    Performs DFT algorithm on the inputs.

    :param inputs: Input Signal of shape: (batch_size x N) (Real or Complex data acceptable, function will cast to
           tf.complex64 regardless).
    :param twiddle_array: Array of Twiddle Factors which is (N x N) in size.
    :param verbose: Boolean value controlling whether or not the function prints notifications as it runs.
    :param return_real:  Boolean value to determine whether or not to perform the tf.abs operation on the DFT output.
    :return y_output/y_prediction: Magnitude DFT of the Input Signal (dtype = tf.float32), or the actual DFT of the
            Input Signal (dtype = tf.complex64), respectively.
    :return grad: Handle to the grad(...) function, which computes the gradient of the Error signal.
    """

    # Checking inputs for validity
    if not tf.is_tensor(inputs):
        # --Changing input to tensor
        if verbose:
            print('Changing input to tensor')
        inputs = tf.convert_to_tensor(inputs, dtype=tf.complex64)

    if not (inputs.dtype == tf.complex64):
        # --Changing input to complex64
        if verbose:
            print('Changing input to complex64')
        inputs = tf.cast(inputs, tf.complex64)

    N = inputs.shape.as_list()[-1]
    assert N is not None 'Signal Length has been read as None'

    # Checking input length
    if not math.log2(N).is_integer():
        # --Changing input length
        if verbose:
            print('Changing input length')
        num_zeros_to_add = next_power_of_2(N) - N
        inputs = tf.concat([inputs, tf.zeros(num_zeros_to_add, dtype=tf.complex64)])

    # Checking twiddle_array for validity
    if twiddle_array is None:
        # --Define the Twiddle Array for this calculation if one is not provided
        if verbose:
            print('Generating Twiddle Array')
        twiddle_array = []
        for i in range(N):
            row = []
            for j in range(N):
                row.append(Wnp(N=N, p=(i * j)))
            twiddle_array.append(row)
        # --Convert to complex64 tensor
        twiddle_array = tf.convert_to_tensor(twiddle_array, dtype=tf.complex64)

    if not tf.is_tensor(twiddle_array):
        # --Changing twiddle array to tensor
        if verbose:
            print('Changing twiddle array to tensor')
        twiddle_array = tf.convert_to_tensor(twiddle_array, dtype=tf.complex64)

    if not (twiddle_array.dtype == tf.complex64):
        # --Changing twiddle array to complex64
        if verbose:
            print('Changing twiddle array to complex64')
        twiddle_array = tf.cast(twiddle_array, dtype=tf.complex64)

    assert N == twiddle_array.shape.as_list()[0] and N == twiddle_array.shape.as_list()[1], \
        'Input tensor and Twiddle Array do not have compatible shapes'

    if verbose:
        print('Computing DFT')
    y_prediction = tf.tensordot(inputs, twiddle_array, axes=1, name='dft_calc')
    y_output = tf.abs(y_prediction, name='dft_mag')

    def grad(dEdy):
        """
        Function computes the gradient of the Error signal w.r.t. the inputs of the dft(...) function.

        :param dEdy: Gradient of the Error signal passed backwards from the next layer up in the network
        :return dEdx: Gradient of the Error w.r.t. the inputs to the DFT layer
        :return dEdW:  Gradient of the Error w.r.t. the Twiddle matrix in the DFT layer
        """

        # dEdx = dydx * dEdy
        # dydx = W
        # Therefore: dEdx = W * dEdy
        dEdx = tf.tensordot(twiddle_array, tf.transpose(dEdy), name='dEdx')

        # dEdW = dydW * dEdy
        # dydW = x
        # Therefore: dEdW = x * dEdy
        dEdW = tf.tensordot(tf.transpose(inputs), dEdy, name='dEdW')

        return dEdx, dEdW, None, None

    if return_real:
        return y_output, grad
    else:
        return y_prediction, grad


class FFT(layers.Layer):
    """
    This layer is designed to initially perform a standard FFT.
    """

    def __init__(self, input_shape, **kwargs):
        super(FFT, self).__init__(**kwargs)
        num_samples = next_power_of_2(input_shape.as_list()[-1])
        self.radius = self.add_weight(shape=(1, ((num_samples // 2) * math.log2(num_samples))),
                                      initializer='ones', trainable=True, dtype=tf.float32)
        self.angle = self.add_weight(shape=(1, ((num_samples // 2) * math.log2(num_samples))),
                                     initializer='zeros', trainable=True, dtype=tf.float32)

    def call(self, inputs, **kwargs):
        output_val = fft_custom(inputs, self.radius, self.angle)
        return output_val

    def get_config(self):
        config = super(FFT, self).get_config()
        config.update({'radius': self.radius, 'angle': self.angle})
        return config


# noinspection PyBroadException
class DFT(layers.Layer):
    """
    This layer is designed to initially perform a standard DFT.
    """

    def __init__(self, input_shape: list or tuple, verbose: bool = False, return_real: bool = True,
                 **kwargs):
        super(DFT, self).__init__(**kwargs)

        num_samples = next_power_of_2(input_shape.as_list()[-1])
        file_number = math.log2(num_samples)

        W = []
        for i in range(num_samples):
            row = []
            for j in range(num_samples):
                row.append(Wnp(N=num_samples, p=(i * j)))
            W.append(row)

        self.twiddle = tf.Variable(initial_value=tf.convert_to_tensor(W, dtype=tf.complex64), trainable=True,
                                   dtype=tf.complex64)
        self.verbose = verbose
        self.return_real = return_real

    def call(self, inputs, **kwargs):
        output_val = dft(inputs, self.twiddle, verbose=self.verbose, return_real=self.return_real)
        return output_val

    def get_config(self):
        config = super(DFT, self).get_config()
        config.update({'twiddle': self.twiddle, 'verbose': self.verbose, 'return_real': self.return_real})
        return config


if __name__ == '__main__':
    """ 
    Simple test function for the FFT and DFT classes
    run like this:
        `python -m Fourier_Transform` 
    """

    font = {'family': 'monospace',
            'weight': 'bold',
            'size': 5}
    rc('font', **font)

    def random_sine_generator(sig_len: int, batch_size: int = 1, plot_data: bool = False):
        fig, axs = plt.subplots(2, 2)
        plt.ion()
        while True:
            batch_samples = np.zeros(shape=(batch_size, sig_len))
            batch_targets = np.zeros(shape=(batch_size, sig_len))
            for i in range(batch_size):
                OSR = 10 * np.random.rand()
                num_cycles = (sig_len + 1) / OSR
                sig_f = 10000000 * np.random.rand()
                sig_phase = (-2*np.pi) + (4*np.pi) * np.random.rand()
                NFFT = sig_len

                t, clean_sig = sine_wave(f=sig_f, overSampRate=OSR, phase=sig_phase, nCyl=num_cycles)
                if not math.log2(len(clean_sig)).is_integer():
                    if len(clean_sig) > signal_length:
                        clean_sig = clean_sig[0:signal_length]
                        t = t[0:signal_length]
                    else:
                        num_zeros_to_append = len(clean_sig) - signal_length
                        clean_sig = np.pad(clean_sig, [0, num_zeros_to_append], mode='constant', constant_values=0)
                        t_delta = t[1] - t[0]
                        for idx in range(num_zeros_to_append):
                            t = np.pad(t, [0, 1], mode='constant', constant_values=t[-1] + t_delta)

                clean_fft = fft(clean_sig, NFFT)
                noisy_sig = clean_sig + np.random.normal(scale=2, size=sig_len)
                noisy_fft = fft(noisy_sig, NFFT)

                if plot_data:
                    plot_signal(axs, t, sig_f, NFFT, clean_sig, noisy_sig, clean_fft, noisy_fft)
                    plt.show()
                    plt.pause(1)
                    axs[0, 0].clear()
                    axs[0, 1].clear()
                    axs[1, 0].clear()
                    axs[1, 1].clear()

                batch_samples[i, :] = noisy_sig
                batch_targets[i, :] = np.abs(clean_fft)

            yield batch_samples, batch_targets


    signal_length = 2 ** 8
    generator = random_sine_generator(signal_length, batch_size=1, plot_data=True)
    input_tensor = Input(shape=(1, signal_length))
    output_layer = DFT(input_shape=input_tensor.shape, name='dft_1')(input_tensor)

    model = Model(inputs=[input_tensor], outputs=[output_layer])
    model.compile(optimizer='rmsprop', run_eagerly=True, loss='mae', metrics=['accuracy'])
    model.summary()

    print('Running generator...\n')
    for a, sample in enumerate(generator):
        if a == 10:
            break
        print(f"Iteration #: {a}")
        # dft_val = output_layer(sample[0])
        # diff = sample[1] - dft_val
        # print(f"Max difference between Generated FFT and layer's output: {np.max(diff)}\n")

    # magnitude, truth = dft(inputs=[1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0], verbose=True)
    # output_fft = fft(inputs=[1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0])

    # input_tensor = Input(shape=(1, signal_length))
    # output_layer = DFT(input_shape=input_tensor.shape, name='dft_1')
    # output = output_layer(input_tensor)
    # model = Model(input_tensor, output)
    #
    # model.compile(loss=losses.mean_squared_error, optimizer='sgd')
    # model.summary()
    #
    # history = model.fit(x=generator, steps_per_epoch=10, epochs=5, validation_data=generator, validation_steps=10,
    #                     verbose=True)
    # print(history)
