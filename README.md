# Explainable Facial Emotion Recognition with the use of Vision Transformers

## Overview
My thesis aims to enhance Facial Emotion Recognition (FER) using Vision Transformer (ViT) architectures. The research focuses on selecting, fine-tuning, and evaluating multiple ViT models to determine the most effective architecture for emotion detection using the FER2013 dataset. The project provides insights into model performance and decision-making processes by employing advanced evaluation metrics and explainability techniques like LIME. Furthermore, the thesis concludes with developing a real-time facial emotion recognition application, which utilizes live video streams from webcams or smartphone cameras to identify facial emotions.

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

## Conclusion
This README provides a comprehensive evaluation of four models trained for facial emotion recognition. Key results such as training performance, classification metrics, confusion matrices, and model explainability using LIME have been included to compare and contrast model behavior. Further refinements could be implemented to optimize real-time performance for facial emotion detection applications.
