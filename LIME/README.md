# LIME (Local Interpretable Model-agnostic Explanations)

The **LIME** technique is used to explain the predictions made by these models. LIME generates interpretable explanations by approximating the model's decision boundary locally around a particular input.

### How LIME Works
1. LIME perturbs the input image by making small changes (e.g., turning segments of the image gray).
2. The perturbed images are passed through the model to observe how these changes affect the model's predictions.
3. The most important regions of the image that influence the model's prediction are highlighted.

## LIME Implementation for Each Model

LIME is implemented for each of the models as follows:

1. **ViTCN.py**: 
    - Implements LIME for the ViTCN model, which processes temporal sequences of image features.
    - The LIME explanations are applied to the final prediction of the emotion classes.

2. **DeiT.py**: 
    - Implements LIME for the DeiT model.
    - LIME highlights the key areas of the image that influence the transformer-based model's classification.

3. **CeiT.py**: 
    - Implements LIME for the CeiT model.
    - The explanations reveal the contribution of both convolutional and transformer layers to the model's decision.

4. **CvT.py**: 
    - Implements LIME for the CvT model.
    - The convolutional and transformer components jointly influence the LIME-based explanation.

## How to Run the LIME Explanations

To run the LIME explanations for any of the models, follow the steps below:

1. **Install Dependencies**:
    Make sure you have all the necessary libraries installed by running:
    ```bash
    pip install -r requirements.txt
    ```
2. **Run the LIME Explanation**:
    Run the LIME implementation for a model using the corresponding Python script:
    ```bash
    python ViTCN.py
    python DeiT.py
    python CeiT.py
    python CvT.py
    ```

3. **View the Explanation**:
    Each script will display two images for each emotion class:
    - The original image.
    - The LIME explanation highlighting the most important regions that contribute to the model's classification decision.

## Explainable Results
- Visual explanation of how the ViTCN model interprets input images when predicting specific emotions.

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/466559a4-56be-400c-b85a-581b4133cb1a" alt="ViTCN Surprise" width="200"></td>
    <td><img src="https://github.com/user-attachments/assets/fd6804c3-44e3-4de8-8311-461e70ac132d" alt="ViTCN Sad" width="200"></td>
    <td><img src="https://github.com/user-attachments/assets/ec71df09-ae45-4b83-8d09-276e2e1cfbac" alt="ViTCN Disgust" width="200"></td>
    <td><img src="https://github.com/user-attachments/assets/02c70507-c7d9-4fe1-93ac-592298151dff" alt="ViTCN Anger" width="200"></td>  
  </tr>
  <tr>
    <td>ViTCN Suprise</td>
    <td>ViTCN Sad</td>
    <td>ViTCN Disgust</td>
    <td>ViTCN Anger</td>
  </tr>
</table>
