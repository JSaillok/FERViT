# Explainable Facial Emotion Recognition with the use of Vision Transformers

## Overview
My thesis aims to enhance Facial Emotion Recognition (FER) using Vision Transformer (ViT) architectures. The research focuses on selecting, fine-tuning, and evaluating multiple ViT models to determine the most effective architecture for emotion detection using the FER2013 dataset. The project provides insights into model performance and decision-making processes by employing advanced evaluation metrics and explainability techniques like LIME. Furthermore, the thesis concludes with developing a real-time facial emotion recognition application, which utilizes live video streams from webcams or smartphone cameras to identify facial emotions.

## Objectives
**Research ViT Architectures:** Investigate and analyze various Vision Transformer architectures, including but not limited to models like hybrid ViTs, to understand their strengths and limitations in the context of facial emotion recognition.

**Fine-Tuning for Diverse Datasets:** Implement fine-tuning techniques to adapt ViT models for diverse facial expression datasets. This involves training the model on datasets with various facial expressions, ensuring robust performance across different scenarios.

**Model Selection:** Select the most suitable ViT architecture for facial emotion recognition based on the evaluation results. This will involve a comparative analysis of different models to identify the most effective approach for the task.

**Model Evaluation:** Evaluate the performance of the fine-tuned ViT models on the fer2013 dataset. This will involve training the models on the dataset and assessing their accuracy, precision, recall, F1 score, confusion matrix, and graphs of training and validation loss.

**Evaluation Metrics:** Develop and implement evaluation metrics to assess the performance of the models on the fer2013 dataset. This will involve quantitative and qualitative analysis to understand the model's strengths and weaknesses.

**Explainability Technique:** Explore the use of the Explainable AI (XAI) technique, LIME, to gain insights into the decision-making process of each model. This will help in understanding the model's reasoning behind its predictions.

**Real-Time Emotion Recognition:** Develop an application that leverages the fine-tuned ViT model to perform real-time facial emotion recognition using live video streams from webcams or smartphone cameras.

## Methodology
**Literature Review:** Conduct an in-depth review of existing literature on Vision Transformers (ViT), facial emotion recognition, and related fields. The literature review will focus on identifying state-of-the-art architectures, explainability techniques, and evaluation metrics, establishing a foundation for selecting and fine-tuning models.

**Model Selection:** Analyze and select the most suitable ViT architectures based on the literature review and the specific requirements of facial emotion recognition. This step will involve a comparative analysis of various ViT models, including ViTCN, DeiT, CeiT, and CvT, to identify strengths and limitations, as well as their potential for real-time application.

**Dataset Preparation:** Gather and preprocess the fer2013 dataset, ensuring it is in a suitable format for training and evaluation. This includes data cleaning, normalization, and splitting the dataset into training, validation, and test sets.

**Fine-tuning the Models:** Experiment with fine-tuning techniques to adapt the selected ViT models for the FER2013 dataset. This process involves adjusting pre-trained models to better capture facial expression nuances, ensuring improved accuracy and robustness across diverse facial expressions.

**Model Evaluation:** Evaluate the fine-tuned models on the FER2013 dataset using key metrics such as accuracy, precision, recall, and F1-score, also use confusion matrix and implement graphs for training and validation behavior. Perform a detailed quantitative and qualitative analysis of the results to determine each model's strengths and weaknesses.

**Explainability with LIME:** Apply the LIME (Local Interpretable Model-Agnostic Explanations) technique to make the decision-making process of the ViT models interpretable. This step will help explain why a particular model classifies an image as a specific emotion, providing transparency and insights into model behavior.

**Real-Time Application Development:** Develop a real-time facial emotion recognition application using the fine-tuned ViT model. The application will integrate live video streams from webcams or smartphone cameras, allowing users to interactively recognize facial emotions. The application will also provide LIME-based explanations for the detected emotions.

## Dataset
The FER2013 dataset is used for training and evaluating the Vision Transformer models. It contains 35,887 labeled facial images representing seven different emotions.

- **Download the dataset:** [Kaggle - FER2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

- **Preprocessing steps:** The images were resized and normalized for optimal model training. Data augmentation techniques such as flipping and rotation were applied to improve model generalization.

## Models
This thesis implements several Vision Transformer models, including:
- **ViTCN**: A hybrid model combining Vision Transformers and Temporal Convolution Networks for sequential image processing.
- **DeiT**: Data-efficient Vision Transformer for high performance with less training data.
- **CeiT**: Convolution-Enhanced Image Transformer designed to improve visual processing with convolutional layers.
- **CvT**: Convolutional Vision Transformer using convolutions for improved spatial structure preservation.

Each model has been fine-tuned for emotion recognition on the FER2013 dataset. You can find the code for each model in the respective folders.

## Explainability with LIME
Explainable AI (XAI) is a critical aspect of this project, aiming to make the decision-making process of Vision Transformers (ViT) transparent and interpretable. This thesis uses LIME (Local Interpretable Model-Agnostic Explanations) to provide insights into how each ViT model makes predictions on facial emotions.

LIME generates explanations by perturbing the input image and observing how the model's predictions change. This process helps visualize which parts of the image contribute most to the model’s decision.

- **Objective:** To enhance the interpretability of the ViT models by providing local explanations for each prediction.
- **Key Insight:** LIME can help identify whether the models focus on relevant facial features (such as eyes, mouth, etc.) when predicting emotions.
- **Visualization:** For each predicted emotion, LIME highlights the regions of the face that influenced the decision.

## Real-Time Application Development
This thesis also integrates the fine-tuned ViT models into a real-time emotion recognition application. Using a webcam or smartphone camera, the system captures live video streams and performs real-time facial emotion detection.

- **Objective:** Develop a practical system for detecting facial emotions in real-time using the fine-tuned ViT models.
- **Technology:** The application uses OpenCV for capturing live video streams, while the ViT models process each frame to detect emotions.
- **Models:** The user can select between different models (DeiT, ViTCN, CeiT, CvT), each with pre-trained weights, and start the real-time emotion detection process.
- **Explainability:** For each prediction, the system can also generate a LIME explanation in real-time to show the user which facial features the model focused on.

## Results Overview
For each model, the following results are presented:
- **Training/Validation Loss and Accuracy Graphs**: These graphs provide insights into model convergence and overfitting.
- **Classification Reports**: Detailed reports showcasing precision, recall, and F1-score across all emotion classes.
- **Confusion Matrices**: Visual representation of true positives, false positives, and false negatives for all emotion categories.
- **LIME Explainability Results**: Visualization and explanation of specific model predictions, showing which parts of the image influenced the model's decision.

---

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
- Confusion matrix visualization of the ViTCN model.
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
- Confusion matrix visualization of the DeiT model.
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
- Confusion matrix visualization of the CeiT model.
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
- Confusion matrix visualization of the CvT model.
![image](https://github.com/user-attachments/assets/c7e3ab54-4966-4205-9b8f-19101e175329)

#### LIME Explainability
- Visual explanation of how the CvT model interprets input images when predicting specific emotions.

---

## Conclusion
This README provides a comprehensive evaluation of four models trained for facial emotion recognition. Key results such as training performance, classification metrics, confusion matrices, and model explainability using LIME have been included to compare and contrast model behavior. Further refinements could be implemented to optimize real-time performance for facial emotion detection applications.
