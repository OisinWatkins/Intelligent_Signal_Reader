# Intelligent_Signal_Reader
This project contains all files pertaining to my attempt to design an intelligent signal reader using Machine Learning

### Generators:
_Generator.py_ was the first file written. This file contains 3 generator definitions: `prepare_data`, 
`batch_file_data` and `read_radar_data`. Together these generators read, prepare and present batches of the .mat files
generated using the RADAR simulator developed in MATLAB and stored on GIT at: 
https://github.com/OisinWatkins/MATLAB_RADAR_Simulator

### Intelligent Fourier Transform:
_Fourier_Transform.py_ contains the code involved in performing either a Discrete Fourier Transform (DFT) or a Fast 
Fourier Transform (FFT) and the definitions of DFT and FFT layers which can be added to a network. These layers
contain very few attributes, the primary attribute being the "twiddle array", which is the active matrix used to 
perform the transform. This matrix is trained over time. 

### Intelligent DFT mlapp:
_Intelligent DFT.mlapp_ is a MATLAB application which is used to demo this idea and provide both a proof of concept and
an active demonstration of this algorithm being used in a Radio reciever. Nothing in this file is meant to be exported to
any final application, it is only a platform to play with training settings and get a feel for how the system behaves.
