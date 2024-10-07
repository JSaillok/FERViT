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

---

## Performance Results

### 1. ViTCN Model Results
#### Training/Validation Graphs
- **Training Accuracy vs. Validation Accuracy**
- **Training Loss vs. Validation Loss**
![image](https://github.com/user-attachments/assets/b878f8ae-1a11-4ca6-94e2-b33442ac4748)

#### Classification Report
| Class    | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Angry    | 0.64      | 0.67   | 0.65     | 491     |
| Disgust  | 0.89      | 0.73   | 0.80     | 55      |
| Fear     | 0.63      | 0.52   | 0.57     | 528     |
| Happy    | 0.91      | 0.89   | 0.90     | 879     |
| Sad      | 0.57      | 0.65   | 0.61     | 594     |
| Surprise | 0.86      | 0.79   | 0.82     | 416     |
| Neutral  | 0.66      | 0.71   | 0.68     | 626     |
| **Accuracy** |       |        | **0.72** | 3589    |
| **Macro Avg** | 0.74 | 0.71   | 0.72     | 3589    |
| **Weighted Avg** | 0.72 | 0.72  | 0.72   | 3589    |

#### Confusion Matrix
![image](https://github.com/user-attachments/assets/19fc7800-bcf5-41ab-afc7-1025b927c7ec)

#### LIME Explainability
- Visual explanation of how the ViTCN model interprets input images when predicting specific emotions.

---

### 2. DeiT Model Results
#### Training/Validation Graphs
- **Training Accuracy vs. Validation Accuracy**
- **Training Loss vs. Validation Loss**
![image](https://github.com/user-attachments/assets/5426645c-3e21-4ae3-9aca-6a414ccc5635)

#### Classification Report
| Class    | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Angry    | 0.66      | 0.63   | 0.64     | 491     |
| Disgust  | 0.91      | 0.73   | 0.81     | 55      |
| Fear     | 0.58      | 0.59   | 0.59     | 528     |
| Happy    | 0.88      | 0.90   | 0.89     | 879     |
| Sad      | 0.60      | 0.56   | 0.58     | 594     |
| Surprise | 0.87      | 0.81   | 0.84     | 416     |
| Neutral  | 0.66      | 0.74   | 0.69     | 626     |
| **Accuracy** |        |        | **0.72** | 3589   |
| **Macro Avg** | 0.74 | 0.71   | 0.72     | 3589    |
| **Weighted Avg** | 0.72 | 0.72   | 0.72   | 3589   |

#### Confusion Matrix
![image](https://github.com/user-attachments/assets/7374c55d-9019-4d7d-979f-055260a2e6db)

#### LIME Explainability
- Visual explanation of how the DeiT model interprets input images when predicting specific emotions.

---

### 3. CeiT Model Results
#### Training/Validation Graphs
- **Training Accuracy vs. Validation Accuracy**
- **Training Loss vs. Validation Loss**
![image](https://github.com/user-attachments/assets/5436a76c-8921-452c-b416-e67b00dd5d8f)

#### Classification Report
| Class    | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Angry    | 0.60      | 0.61   | 0.60     | 491     |
| Disgust  | 0.73      | 0.65   | 0.69     | 55      |
| Fear     | 0.55      | 0.50   | 0.52     | 528     |
| Happy    | 0.90      | 0.86   | 0.88     | 879     |
| Sad      | 0.53      | 0.59   | 0.56     | 594     |
| Surprise | 0.80      | 0.81   | 0.80     | 416     |
| Neutral  | 0.68      | 0.69   | 0.68     | 626     |
| **Accuracy** |        |        | **0.69** | 3589   |
| **Macro Avg** | 0.68 | 0.67   | 0.68     | 3589    |
| **Weighted Avg** | 0.69 | 0.69   | 0.69   | 3589   |

#### Confusion Matrix
![image](https://github.com/user-attachments/assets/afbe275b-3e68-46aa-90d7-e04941cf8c13)

#### LIME Explainability
- Visual explanation of how the CeiT model interprets input images when predicting specific emotions.

---

### 4. CvT Model Results
#### Training/Validation Graphs
- **Training Accuracy vs. Validation Accuracy**
- **Training Loss vs. Validation Loss**
![image](https://github.com/user-attachments/assets/babc2ee2-51b1-4a3a-8c7a-96c1333ab1b4)

#### Classification Report
| Class    | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Angry    | 0.63      | 0.60   | 0.62     | 491     |
| Disgust  | 0.26      | 0.82   | 0.40     | 55      |
| Fear     | 0.60      | 0.51   | 0.55     | 528     |
| Happy    | 0.94      | 0.87   | 0.90     | 879     |
| Sad      | 0.56      | 0.57   | 0.57     | 594     |
| Surprise | 0.79      | 0.85   | 0.82     | 416     |
| Neutral  | 0.66      | 0.67   | 0.67     | 626     |
| **Accuracy** |        |        | **0.69** | 3589   |
| **Macro Avg** | 0.64 | 0.70   | 0.65     | 3589    |
| **Weighted Avg** | 0.71 | 0.69   | 0.70   | 3589   |

#### Confusion Matrix
![image](https://github.com/user-attachments/assets/c7e3ab54-4966-4205-9b8f-19101e175329)

#### LIME Explainability
- Visual explanation of how the CvT model interprets input images when predicting specific emotions.
