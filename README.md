# Intelligent_Signal_Reader
This project contains all files pertaining to my attempt to design an intelligent signal reader using Machine Learning

### Generators:
_Generator.py_ was the first file written. This file contains 3 generator definitions: `prepare_data`, 
`batch_file_data` and `read_radar_data`. Together these generators read, prepare and present batches of the .mat files
generated using the RADAR simulator developed in MATLAB and stored on GIT at: 
https://github.com/OisinWatkins/MATLAB_RADAR_Simulator

### Simple Networks:
_Simple_Dense_Network.py_, _Conv_Network.py_, _Highway_Network.py_, _Divergent_Network.py_ and _Neural_Net_Theano.py_ 
are all experiments in viewing the typical network layouts' responses to complex RADAR data. As expected, no 
network achieved much accuracy, given that the input data is so far removed from the output data.

### Intelligent Fourier Transform:
_Fourier_Transform.py_ contains the code involved in performing either a Discrete Fourier Transform (DFT) or a Fast 
Fourier Transform (FFT) and the definitions of DFT and FFT layers which can be added to a network. These layers
contain very few attributes, the primary attribute being the "twiddle array", which is the active matrix used to 
perform the transform. This matrix is trained over time. 
