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

        kernel_regularizer: Regulariser function applied to
            the `twiddle` weights matrix.

        kernel_constraint: Constraint function applied to
            the `twiddle` weights matrix.

    Input shape:
        N-D tensor with shape: `(batch_size, num_samples)`.

    Output shape:
        N-D tensor with shape: `(batch_size, num_samples)`.
    """

    def __init__(self, num_samples: int, twiddle_initialiser=None, kernel_regularizer=None, kernel_constraint=None, **kwargs):
        super(DFT, self).__init__(**kwargs)

        if (num_samples is None) or not (num_samples > 0):
            raise ValueError('The dimension of the inputs to `DFT` should be defined. Found `None` or `0`.')
            
        if not math.log2(num_samples).is_integer():
            # --Changing num_samples
            print(f" -> Brining num_samples from current value: {num_samples} to the next power of 2: {next_power_of_2(num_samples)}\n")
            num_samples = next_power_of_2(num_samples)


        if twiddle_initialiser is None:
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


if __name__ == '__main__':
    """
    This main function is here to be used as a quick explanation of how these layers work.
    Run like this:
        `python -m Fourier_Transform`
        
    The DFT layer takes an input which is either real or imaginary. As part of it's forward pass
    it will cast the input to tf.complex64 regardless. The output will be the unprocessed DFT of the
    input. It is advisable to post-process the DFT output into real-valued data before feeding it
    to another layer. There does tend to be an issue in the first epoch when the input has length
    zero, but for every epoch thereafter there are no issues. 
    
    Be sure to carefully research the optimizer and loss function used with this layer, as it may be 
    that not all optimizers and loss functions are compatible with this layer. 
    """
    model = Sequential()
    model.add(tf.keras.layers.Input(shape=(256)))
    model.add(DFT(num_samples=256))

    model.compile(optimizer='rmsprop', loss='mae')

    print("\n>>\n"
          ">> Below will be defined a very simple model to show you what to expect when you\n"
          ">> include this layer in your models. It's best (where possible) to follow the below\n"
          ">> guidelines in your projects:\n"
          ">> -> Keep your input of length 2^n to avoide padding the input signal with 0's\n"
          ">> -> Have this layer high up in your model, close to the input.\n"
          ">> -> Follow this layer with post-processing which will convert the DFT to meaningful real-valued data.\n"
          ">> -> Make certain your optimizer and loss function (and all other configurables) are compatible with the "
          "layer.\n"
          ">>\n")

    model.summary()

    print("\n>>\n"
          ">> Note the number of parameters. The DFT layer will have 2 * (N ^ 2) parameters\n"
          ">> for an input signal of length N, all of which are trainable. It is also a slow layer to initialise,\n"
          ">> though it is not normally too slow in processing.\n"
          ">>\n"
          ">> The first epoch with this layer will usually flag an issue with the signal\n"
          ">> length (N) being measured as `None`. I haven't yet identified the root cause of \n"
          ">> this issue, but it does not seem to cause any major distress to the system.\n"
          ">>\n"
          ">> To give a bit more of an idea as to what a more complicated model might look like,\n"
          ">> here's a slightly deeper network made using the Keras functional API.\n"
          ">>\n")

    model = None

    signal = Input(shape=(128))
    dft_output = DFT(num_samples=128)(signal)

    dft_output_mag = tf.abs(dft_output)

    dense_1 = layers.Dense(units=64, activation='relu')(dft_output_mag)
    output = layers.Dense(1)(dense_1)

    model = Model(signal, output)
    model.compile(optimizer='rmsprop', loss='mae')
    print("\n")
    model.summary()

    print("\n>>\n"
          ">> As mentioned before, having all configurations of your models be compatible with the DFT\n"
          ">> layer will be essential. To serve as a starting point, note that in my models I used:\n"
          ">>\n"
          ">> -> model.compile(optimizer='rmsprop', loss='mae')\n"
          ">>\n"
          ">> This configuration is known to work. I'm sure other configurations will work, I have\n"
          ">> not tested for them yet.\n"
          ">>\n"
          ">> The only thing left to say is best of luck working with these layers! I hope they help\n"
          ">> with your data science needs!\n"
          ">>")
    
    model = None
