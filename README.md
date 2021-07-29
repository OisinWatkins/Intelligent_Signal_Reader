# Intelligent_Signal_Reader
This project contains all files pertaining to my attempt to design an intelligent signal reader using Machine Learning

### Primary Code Base
_Fourier_Transform.py_ contains the code involved in performing a Discrete Fourier Transform (DFT)
and the definition of DFT layer which can be added to a network. This layer contains very few attributes,
the primary attribute being the "twiddle array", which is the active matrix used to 
perform the transform. This matrix is trained over time.

_DFT_Model_Demo.py_ contains code for the definition and training of a full model which uses the 
DFT Layer to process 1-second audio inputs before processing the resulting tensor using 1-D convnets.
Multiple model topographies were attempted, of each topography only the best performing model was saved.
All training attempts have had their printed output saved for reference

### Misc

_Intelligent DFT.docx_ is the paper which documents all the work done in this project, from the MATLAB proof of concept to the
latest Python implementation.

_Diagrams.ppt_ is the PowerPoint presentation where I designed all the diagams used in the Intelligent DFT.docx paper.

#### Demonstration
Contains some MATLAB code, a MATLAB Application, as well as a collection of images used in the demonstration of this code.

_Intelligent DFT.mlapp_ is a MATLAB application which is used to demo this idea and provide both a proof of concept and
an active demonstration of this algorithm being used in a Radio reciever. Nothing in this file is meant to be exported to
any final application, it is only a platform to play with training settings and get a feel for how the system behaves.

#### Models
Contains a number of .txt files, .png files and a few .h5 files all pertaining the the training attempts conducted as experiments
for the Layer designed in the Primary Code Base.

####Referneces
Houses all the Literature used as reference for this project, as well as a few presentations I wrote to accompany this work.
