# Emotion Recognition Models on FER2013

This folder contains implementations of models for facial emotion recognition using the FER2013 dataset. Each python file follows a standard process for data preparation, model initialization, training, and evaluation.

### Preprocessing

In each file, the dataset is preprocessed to resize the images, apply normalization, and augment the data with techniques such as random flipping and rotation.

### Models Overview

1. **ViTCN (Vision Transformer with Temporal Convolution Network)**
The **ViTCN** model combines the Vision Transformer architecture with a Temporal Convolution Network (TCN) to classify facial emotions. It processes images from the FER2013 dataset and performs emotion classification across seven categories: Anger, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

- Pretrained on: `google/vit-base-patch16-224`.
- Custom TCN layers to process the feature embeddings.
- Model output: Seven emotion categories.
  
2. **DeiT (Data-efficient Image Transformer)**
The **DeiT** model is a Vision Transformer architecture designed for efficiency. It is adapted for the FER2013 dataset to classify facial emotions. The transformer uses attention mechanisms to capture global information from facial features.

- Pretrained on: `facebook/deit-base-patch16-224`.
- Fine-tuned for FER2013.
  
3. **CeiT (Convolution-Enhanced Image Transformer)**
The **CeiT** model introduces convolutional layers into the Vision Transformer to capture both local and global information from images. This hybrid architecture improves the model's ability to classify facial emotions accurately from the FER2013 dataset.

- Pretrained on: `CeiT-Tiny`.
  
4. **CvT (Convolutional Vision Transformer)**
The **CvT** model is a hybrid architecture that combines convolutions and transformers. This model leverages the benefits of convolution for local feature extraction and transformers for long-range dependencies, making it suitable for facial emotion recognition.

- Pretrained on: `CvT-13`.

**Each file contains**:
- Dataset Processing: Code to load and preprocess FER2013.
- Model Initialization: Code to initialize the corresponding model.
- Training: Steps to train the model on FER2013.
- Evaluation: Model evaluation using accuracy, F1-score, confusion matrix, and visual results.

### Installation
To run any file, make sure the following dependencies are installed:

```bash
pip install -r requirements.
```

For CeiT implementation, clone the repository: https://github.com/coeusguo/ceit.git, unzip ceit-main, and copy the ceit folder to the current directory.

```bash
unzip ceit-main.zip
cp -r ceit-main/ceit <path_to_your_project_directory>
```

### Training Tips:
- Ensure that you have a GPU enabled to speed up the training process.
- Adjust the learning rate and batch size depending on the model and hardware capabilities.
- Use early stopping to prevent overfitting and save the best model during training.