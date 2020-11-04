"""
Author: Oisín Watkins
Email: oisinwatkins97@gmail.com
"""

import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.python.keras import constraints
from tensorflow.python.keras import regularizers


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


class DFT(layers.Layer):
    """
    This layer implements a DFT on the input signal.

    output = input * tf.complex(twiddle_real, twiddle_imag)

    where `input` is the input signal of length N and and `twiddle_real` / `twiddle_imag`
    is a matrix created by the layer of size N x N. Each index in `twiddle` is half of a
    Twiddle Factor (dtype: tf.complex64) needed to perform the DFT, and is generated using the
    Wnp function defined in this file. The output of this layer is computed using the
    tf.tensordot(...) function.

    Arguments:
        num_samples: Number of samples in the input signal.
        
        twiddle_initialiser: A numpy array which holds the
            default twiddle array for the users' application.

        kernel_regularizer: Regulariser function applied to
            the `twiddle` weights matrix.

        kernel_constraint: Constraint function applied to
            the `twiddle` weights matrix.

    Input shape:
        N-D tensor with shape: `(batch_size, num_samples)`.

    Output shape:
        N-D tensor with shape: `(batch_size, num_samples)`.
    """

    def __init__(self, num_samples: int=None, twiddle_initialiser=None, kernel_regularizer=None, kernel_constraint=None, **kwargs):
        super(DFT, self).__init__(**kwargs)

        if twiddle_initialiser is None:
            if (num_samples is None) or not (num_samples > 0):
                raise ValueError('The dimension of the inputs to `DFT` should be defined. Found `None` or `0`.')
            
            if not math.log2(num_samples).is_integer():
                # --Changing num_samples
                print(f" -> Brining num_samples from current value: {num_samples} to the next power of 2: {next_power_of_2(num_samples)}\n")
                num_samples = next_power_of_2(num_samples)
                
            W = []
            for i in range(num_samples):
                row = []
                for j in range(num_samples):
                    row.append(Wnp(N=num_samples, p=(i * j)))
                W.append(row)
                
        else:
            W = twiddle_initialiser

        W = tf.convert_to_tensor(W, dtype=tf.complex64)

        self.twiddle_real = tf.Variable(initial_value=tf.math.real(W),
                                        trainable=True, dtype=tf.float32, name='twiddle_real')
        self.twiddle_imag = tf.Variable(initial_value=tf.math.imag(W),
                                        trainable=True, dtype=tf.float32, name='twiddle_imag')
        W = None
        
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.built = True

    def call(self, inputs, **kwargs):
        # Checking inputs for validity
        if not tf.is_tensor(inputs):
            # --Changing input to tensor
            inputs = tf.convert_to_tensor(inputs, dtype=tf.complex64)

        if not (inputs.dtype == tf.complex64):
            # --Changing input to complex64
            inputs = tf.cast(inputs, tf.complex64)

        N = inputs.shape.as_list()[-1]
        if N is None:
            print('Signal Length has been read as None')
            N = self.twiddle_real.shape.as_list()[0]
            print(f"N changed to {N}")

        # Checking input length
        if not math.log2(N).is_integer():
            # --Changing input length
            num_zeros_to_add = next_power_of_2(N) - N
            inputs = tf.concat([inputs, tf.zeros(num_zeros_to_add, dtype=tf.complex64)])

        if not (N == self.twiddle_real.shape.as_list()[0] and N == self.twiddle_real.shape.as_list()[1]) \
                and not (N == self.twiddle_imag.shape.as_list()[0] and N == self.twiddle_imag.shape.as_list()[1]):
            print(f"Input tensor and Twiddle Array do not have compatible shapes\nInput Tensor shape: "
                  f"{inputs.shape.as_list()}\nTwiddle Array Shape: {self.twiddle.shape.as_list()}")
            raise ValueError("Input shape is invalid and/or Twiddle array shape is invalid")

        output_val = tf.tensordot(inputs, tf.complex(self.twiddle_real, self.twiddle_imag), axes=1, name='dft_calc')
        return output_val

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(DFT, self).get_config()

        config.update({'twiddle_real': self.twiddle_real.numpy(),
                       'twiddle_imag': self.twiddle_imag.numpy(),
                       'kernel_regularizer': self.kernel_regularizer,
                       'kernel_constraint': self.kernel_constraint
                       })
        return config
