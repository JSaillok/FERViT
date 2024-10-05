# Emotion Recognition Models on FER2013

This folder contains implementations of multiple models for facial emotion recognition using the FER2013 dataset. Each notebook follows a standard process for data preparation, model initialization, training, and evaluation.

## Dataset: FER2013

The FER2013 dataset is a large-scale dataset for facial emotion recognition. It consists of 48x48 grayscale images of faces with 7 emotion labels.

### Preprocessing

In each notebook, the dataset is preprocessed to resize the images, apply normalization, and augment the data with techniques such as random flipping and rotation.

### Models

1. **ViTCN** - A hybrid Vision Transformer model using Temporal Convolutional Networks (TCNs) for facial emotion recognition.
2. **DeiT** - Data-efficient image Transformer model adapted for facial emotion recognition.
3. **CeiT** - Convolution-enhanced image Transformer model using convolutional layers for improved feature extraction.
4. **CvT** - A convolutional vision transformer with hierarchical architecture to capture both local and global information.

Each notebook contains:
- **Dataset Processing**: Code to load and preprocess FER2013.
- **Model Initialization**: Code to initialize the corresponding model.
- **Training**: Steps to train the model on FER2013.
- **Evaluation**: Model evaluation using accuracy, F1-score, confusion matrix, and visual results.

### Installation

To run any notebook, make sure the following dependencies are installed:

```bash
pip install -r requirements.txt
