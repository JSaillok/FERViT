# Real Time Emotion Recognition Application

The folder of "RTR" contains the code for the Real Time Emotion Recognition Application. It is a Python application that uses the OpenCV library to capture video from a webcam, Pytorch to load a pre-trained models, and the TKinter library to create a graphical user interface.

## Methodology

The application uses the models that initiated in the thesis to recognize emotions. After the training and evaluation of the models, the application takes the .pth files of the models and loads them into the application.

## Installation

To run the application, you will need to have Python 3.9 or higher installed on your system. You will also need to install the following Python packages using the following command:
```pip install -r requirements.txt```

***Note that some system-specific libraries like CUDA for GPU support might need to be installed separately, depending on your setup and requirements.***

Also to run the CeiT model, you will need to clone the [repository](https://github.com/coeusguo/ceit), unzip it with the name of "ceit_repo" and place it in the same folder as the "RTR" folder.