"""

The data used for this comparrison was pulled from:

"""

import numpy as np
import tensorflow as tf
from scipy.io import wavfile
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    test_audio_path = ''

    print("\n>>\n"
            ">> Let us read the audio waves and use the below-preprocessing\n"
            ">> steps to deal with the disparity in presentation length. Here\n"
            ">> are the two steps we’ll follow:\n"
            ">>\n"
            ">> -> Resampling\n"
            ">> -> Removing commands shorter than 1 second\n"
            ">>\n")

    all_wave = []
    waves = [f for f in os.listdir(train_audio_path) if f.endswith('.wav')]
    for wav in waves:
        samples, sample_rate = librosa.load(train_audio_path + wav, sr=16000)
        samples = librosa.resample(samples, sample_rate, 8000)
        if (len(samples) == 8000):
            all_wave.append(samples)
            
        if (len(all_wave) == 10000):
            break
            
    # Convert the output labels to integer encoded
    le = LabelEncoder()
    y = le.fit_transform(all_label)
    classes = list(le.classes_)

    y = tf.keras.utils.to_categorical(y, num_classes=len(labels))

    all_wave_dft = np.array(all_wave)
    all_wave_standard = np.array(all_wave).reshape(-1, 8000, 1)