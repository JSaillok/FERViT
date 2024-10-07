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
1. ViTCN model.
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/466559a4-56be-400c-b85a-581b4133cb1a" alt="ViTCN Surprise" width="400"></td>
    <td><img src="https://github.com/user-attachments/assets/fd6804c3-44e3-4de8-8311-461e70ac132d" alt="ViTCN Sad" width="400"></td>
  </tr>
  <tr>
    <td>ViTCN Suprise</td>
    <td>ViTCN Sad</td>
  </tr>

  <tr>
    <td><img src="https://github.com/user-attachments/assets/ec71df09-ae45-4b83-8d09-276e2e1cfbac" alt="ViTCN Disgust" width="400"></td>
    <td><img src="https://github.com/user-attachments/assets/02c70507-c7d9-4fe1-93ac-592298151dff" alt="ViTCN Anger" width="400"></td>  
  </tr>
  <tr>
    <td>ViTCN Disgust</td>
    <td>ViTCN Anger</td>
  </tr>
</table>

2. DeiT model.
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/126d9e6a-b71e-47fc-9b51-2de5e9977004" alt="DeiT Sad" width="200"></td>
    <td><img src="https://github.com/user-attachments/assets/3a9191d2-5765-4726-90df-95813f1b2602" alt="DeiT Surprise" width="200"></td>
    <td><img src="https://github.com/user-attachments/assets/35b41845-5c5d-42d9-bbbc-306a8fa8050c" alt="DeiT Neutral" width="200"</td>
  </tr>
  <tr>
    <td>DeiT Sad</td>
    <td>DeiT Surprise</td>
    <td>DeiT Neutral</td>
  </tr>

  <tr>
    <td><img src="https://github.com/user-attachments/assets/3e90d296-aee0-43c2-b789-3c1f16e315b1" alt="DeiT Fear" width="400"></td>
    <td><img src="https://github.com/user-attachments/assets/a40fed72-98bb-44d0-bc1c-b39b7b0637bd" alt="DeiT Disgust" width="400"></td>  
  </tr>
  <tr>
    <td>DeiT Fear</td>
    <td>DeiT Disgust</td>
  </tr>
</table>

3. CeiT model.
![Ceit_Surprise](https://github.com/user-attachments/assets/db1dae4b-50e8-4250-933d-e5c3d16446e9)
![Ceit_Neutral](https://github.com/user-attachments/assets/c4d05af1-9f9b-4f8d-a96f-6dcfee9b113d)
![Ceit_Happy](https://github.com/user-attachments/assets/d5da53d1-67b2-4bd4-9637-75cc190ba898)
![Ceit_fear](https://github.com/user-attachments/assets/38594ee5-ff03-4e84-b36e-2e5f968fa089)
![Ceit_Disgust](https://github.com/user-attachments/assets/cb0f625a-d031-4b04-a28a-76ce60e71866)
![Ceit_Anger](https://github.com/user-attachments/assets/aef402da-af33-47c7-832e-4cce04d7247b)

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/466559a4-56be-400c-b85a-581b4133cb1a" alt="ViTCN Surprise" width="400"></td>
    <td><img src="https://github.com/user-attachments/assets/fd6804c3-44e3-4de8-8311-461e70ac132d" alt="ViTCN Sad" width="400"></td>
  </tr>
  <tr>
    <td>ViTCN Suprise</td>
    <td>ViTCN Sad</td>
  </tr>

  <tr>
    <td><img src="https://github.com/user-attachments/assets/ec71df09-ae45-4b83-8d09-276e2e1cfbac" alt="ViTCN Disgust" width="400"></td>
    <td><img src="https://github.com/user-attachments/assets/02c70507-c7d9-4fe1-93ac-592298151dff" alt="ViTCN Anger" width="400"></td>  
  </tr>
  <tr>
    <td>ViTCN Disgust</td>
    <td>ViTCN Anger</td>
  </tr>
</table>

4. CvT model.
![Cvt_Anger](https://github.com/user-attachments/assets/6d36e61b-a3d3-4650-985d-d8b5a932018b)
![Cvt_Disgust](https://github.com/user-attachments/assets/7293ca44-47c8-4f8a-8308-b34f3d3b221b)
![Cvt_Fear](https://github.com/user-attachments/assets/2a87a474-c7f0-4227-9a54-ff7f573919ad)
![Cvt_Sad](https://github.com/user-attachments/assets/83d4f506-14fc-42e1-9bbe-bed2abc3cd7a)

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/466559a4-56be-400c-b85a-581b4133cb1a" alt="ViTCN Surprise" width="400"></td>
    <td><img src="https://github.com/user-attachments/assets/fd6804c3-44e3-4de8-8311-461e70ac132d" alt="ViTCN Sad" width="400"></td>
  </tr>
  <tr>
    <td>ViTCN Suprise</td>
    <td>ViTCN Sad</td>
  </tr>

  <tr>
    <td><img src="https://github.com/user-attachments/assets/ec71df09-ae45-4b83-8d09-276e2e1cfbac" alt="ViTCN Disgust" width="400"></td>
    <td><img src="https://github.com/user-attachments/assets/02c70507-c7d9-4fe1-93ac-592298151dff" alt="ViTCN Anger" width="400"></td>  
  </tr>
  <tr>
    <td>ViTCN Disgust</td>
    <td>ViTCN Anger</td>
  </tr>
</table>
