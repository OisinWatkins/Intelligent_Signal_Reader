import math
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model, losses, Input

# tf.enable_eager_execution()
# tfe = tf.contrib.eager


def next_power_of_2(x):
    """
    Given any input value, x, next_power_of_2 returns the next power of 2 above the input

    :param x: Input integer
    :return: The next integer power of 2 above the input value (if x is already an integer power of 2 return x)
    """
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def Wnp(N, p):
    """
    Function makes one Twiddle Factor needed for the FFT algorithm

    :param N: Length of the Fourier Transform input sequence
    :param p: root number for this particular twiddle factor
    :return: twiddle factor: e ^ -j * ((2 * pi * p) / N)
    """
    return tf.math.exp(tf.multiply(tf.complex(0.0, -1.0), tf.complex((2 * np.pi * p / N), 0.0)))


@tf.custom_gradient
def fft(inputs, tuning_radii=None, tuning_angles=None):
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
def dft(inputs, twiddle_array, tuning_radii=None, tuning_angles=None):
    """
    Performs DFT algorithm on the inputs using the tuning_radii and tuning_angles to tune the twiddle array input.

    :param inputs: Input Signal (Real or Complex data acceptable, function will cast to Complex64 regardless)
    :param twiddle_array: Array of Twiddle Factors which is N x N in size. Must be of dtype Complex64
    :param tuning_radii: Tensor from the Layer object, used to tune the magnitude of each twiddle factor
    :param tuning_angles: Tensor from the Layer object, used to tune the angle of each twiddle factor
    :return: DFT of the Input Signal (dtype = tf.float32)
    """

    print('Checking input type')
    if not tf.is_tensor(inputs):
        print('--Changing input to tensor')
        inputs = tf.convert_to_tensor(inputs, dtype=tf.complex64)

    if not (inputs.dtype == tf.complex64):
        print('--Changing input to complex64')
        inputs = tf.cast(inputs, tf.complex64)

    N = inputs.shape.as_list()[-1]
    if N is not None:
        print('Checking input length')
        if not math.log2(N).is_integer():
            print('--Changing input length')
            num_zeros_to_add = next_power_of_2(N) - N
            inputs = tf.concat([inputs, tf.zeros(num_zeros_to_add, dtype=tf.complex64)])

    if tuning_radii is not None and tuning_angles is not None:
        """ twiddle_array = (radii * e ^ -j*angles) * twiddle_array """
        print('Applying Weights')
        twiddle_array = tf.multiply(tf.multiply(tf.complex(tuning_radii, 0.0), tf.math.exp(tf.multiply(tf.complex(0.0, -1.0), tf.complex(tuning_angles, 0.0)))), twiddle_array)

    print('Performing DFT')
    """ return = | twiddle_array . inputs | """
    print('Shape of input: ', inputs.shape.as_list())
    ret = tf.abs(tf.tensordot(twiddle_array, inputs, axes=1), name='dft_calc')
    print('Shape of output: ', ret.shape.as_list())

    def grad(dy):
        return tf.abs(tf.tensordot(twiddle_array, tf.cast(dy, dtype=tf.complex64), axes=1), name='dy/dx'), dy, dy

    return ret, grad


# output_dft = dft(inputs=[1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0], twiddle_array=W, tuning_radii=Rad, tuning_angles=Ang)
# output_fft = fft(inputs=[1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0])


class FFT1D(layers.Layer):
    """
    This layer is designed to initially perform a standard 1D FFT. As the layer trains it will (hopefully) learn
    to keep noise out of the spectrum
    """

    def __init__(self, input_shape, **kwargs):
        super(FFT1D, self).__init__(**kwargs)
        num_samples = next_power_of_2(input_shape.as_list()[-1])
        self.radius = self.add_weight(shape=(1, ((num_samples // 2) * math.log2(num_samples))),
                                      initializer='ones', trainable=True, dtype=tf.float32)
        self.angle = self.add_weight(shape=(1, ((num_samples // 2) * math.log2(num_samples))),
                                     initializer='zeros', trainable=True, dtype=tf.float32)

    def call(self, inputs, **kwargs):
        output_val = fft(inputs, self.radius, self.angle)
        return output_val

    def get_config(self):
        config = super(FFT1D, self).get_config()
        config.update({'radius': self.radius, 'angle': self.angle})
        return config


class DFT1D(layers.Layer):
    """
    This layer is designed to initially perform a standard 1D DFT. As the layer trains it will (hopefully) learn
    to keep noise out of the spectrum
    """

    def __init__(self, input_shape, **kwargs):
        super(DFT1D, self).__init__(**kwargs)
        num_samples = next_power_of_2(input_shape.as_list()[-1])
        self.radius = self.add_weight(shape=(num_samples, num_samples), initializer='ones', trainable=True,
                                      dtype=tf.float32)
        self.angle = self.add_weight(shape=(num_samples, num_samples), initializer='zeros', trainable=True,
                                     dtype=tf.float32)
        W = []
        for i in range(num_samples):
            row = []
            for j in range(num_samples):
                row.append(Wnp(N=num_samples, p=(i * j)))
            W.append(row)

        self.twiddle = tf.convert_to_tensor(W, dtype=tf.complex64)

    def call(self, inputs, **kwargs):
        output_val = dft(inputs, self.twiddle, self.radius, self.angle)
        return output_val

    def get_config(self):
        config = super(DFT1D, self).get_config()
        config.update({'radius': self.radius, 'angle': self.angle, 'twiddle': self.twiddle})
        return config


if __name__ == '__main__':
    """ 
    Simple test function for the FFT1D class
    run like this:
        `python -m Fourier_Transform` 
    """


    def random_sine_generator(batch_size=5):
        x = np.linspace(0, 100, 2 ** 4)
        while True:
            batch_samples = np.zeros(shape=(batch_size, len(x)))
            batch_targets = np.zeros(shape=(batch_size, len(x)))
            for i in range(batch_size):
                clean_sig = (1 + 10 * np.random.random()) * np.sin(x + np.random.random())
                clean_fft = np.fft.fft(clean_sig)
                noisy_sig = clean_sig + np.random.normal(scale=0.1, size=len(x))

                batch_samples[i, :] = noisy_sig
                batch_targets[i, :] = np.abs(clean_fft)

            yield batch_samples, batch_targets


    generator = random_sine_generator(batch_size=1)
    eg_sig = np.linspace(0, 100, 2 ** 4)

    input_tensor = Input(shape=len(eg_sig))
    output_layer = DFT1D(input_shape=input_tensor.shape, name='dft_1d')
    output = output_layer(input_tensor)
    model = Model(input_tensor, output)

    # for a, sample in enumerate(random_sine_generator()):
    #     if a == 10:
    #         break
    #     print(a)
    #     output = output_layer(inputs=sample[0][0][0])
    #     print(output)

    model.compile(loss=losses.mean_absolute_error, optimizer='sgd')
    model.summary()

    history = model.fit(x=generator, steps_per_epoch=10, epochs=5, validation_data=generator, validation_steps=10,
                        verbose=True)
    print(history)
