import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.python.keras import constraints
from tensorflow.python.keras import regularizers
from Fourier_Transform import DFT

if __name__ == '__main__':
    print("\n\t---------    Running Demo for the DFT Layer    ---------\n")

    print("\n>>\n"
          ">> This file will run a bespoke model to handle the following task:\n"
          ">> -> ``\n"
          ">>\n"
          ">> To accomplish this, I will define 2 models:\n"
          ">> -> One with the DFT Layer high up in the architecture.\n"
          ">> -> One using more typical Machine Learning Practices.\n"
          ">>\n"
          ">> After training has completed I will run each model through an extensive test\n"
          ">> to determine whether or not the DFT layer bears any benefit to signal processing networks.\n"
          ">>\n")
