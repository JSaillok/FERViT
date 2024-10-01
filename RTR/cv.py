import sys
sys.path.append('/home/jsaillok/Documents/GitHub/FERViT/RTR/ceit_repo/ceit-main/')
from ceit_model import ceit_small_patch16_224

import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import torch
from torchvision import transforms
import torch.nn.functional as F
from transformers import CvtModel, AutoImageProcessor
import torch
from transformers import ViTModel
import torch.nn as nn

import mediapipe as mp

from torch.cuda.amp import autocast

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the emotion categories (adjust according to your model's labels)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Define the transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Class for ViTCN-based Facial Emotion Recognition (FER)
class ViTCN(nn.Module):
    def __init__(self, num_classes=7):
        super(ViTCN, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.vit.config.num_labels = num_classes
        
        # Replace the classifier head
        self.vit.classifier = nn.Identity()

        # TCN configuration
        self.tcn = nn.Sequential(
            nn.Conv1d(in_channels=768, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            *[nn.Sequential(
                nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(0.3)
            ) for _ in range(7)]
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.vit(x).last_hidden_state
        x = x.permute(0, 2, 1)  # Change to (batch, channels, seq_len)
        x = self.tcn(x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        x = self.fc(x)
        return x

# Class for CvT-based Facial Emotion Recognition (FER)
class CvtForFER(nn.Module):
    def __init__(self, num_classes=7):
        super(CvtForFER, self).__init__()
        self.cvt = CvtModel.from_pretrained('microsoft/cvt-13')
        self.fc = nn.Linear(384, 128)  # Additional fully connected layer
        self.relu = nn.ReLU()          # ReLU activation
        self.classifier = nn.Linear(128, num_classes)  # Final classification layer
    
    def forward(self, pixel_values):
        outputs = self.cvt(pixel_values=pixel_values)
        x = outputs.cls_token_value.squeeze(1)  # cls_token_value has the embeddings from CvT-13
        x = self.fc(x)
        x = self.relu(x)
        logits = self.classifier(x)
        return logits

# Function to load the desired model based on user selection
def load_model(model_name):

    if model_name == 'ViTCN':
        model = ViTCN(num_classes=7)
        model.load_state_dict(torch.load('/home/jsaillok/Documents/GitHub/FERViT/RTR/ViTCN_model.pth', map_location=torch.device('cpu')))
    elif model_name == "DeiT":
        model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
        model.load_state_dict(torch.load('/home/jsaillok/Documents/GitHub/FERViT/RTR/Deit_model.pth', map_location=torch.device('cpu')))
    elif model_name == "CeiT":
        model = ceit_small_patch16_224(pretrained=True)
        model.load_state_dict(torch.load('/home/jsaillok/Documents/GitHub/FERViT/RTR/Ceit_model.pth', map_location=torch.device('cpu')))
    elif model_name == "CvT":
        model = CvtForFER()  # Initialize CvT model
        model.load_state_dict(torch.load('/home/jsaillok/Documents/GitHub/FERViT/RTR/CvT_model.pth', map_location=torch.device('cpu')))

    model.eval()
    model.to(device)
    return model

# Function to preprocess images for CvT
def preprocess_cvt_images(images):
    image_processor = AutoImageProcessor.from_pretrained('microsoft/cvt-13')
    inputs = image_processor(images, return_tensors="pt", do_rescale=False)
    return inputs['pixel_values'].to(device)

# Function to start the webcam feed and perform facial emotion recognition
def start_recognition(selected_model_name):
    model = load_model(selected_model_name)

    # Initialize the face detector
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)
    cap = cv2.VideoCapture(1)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces using Mediapipe
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                face = frame[y:y+h, x:x+w]

                if selected_model_name == "CvT":
                    face_tensor = preprocess_cvt_images(face)
                else:
                    face_tensor = transform(face).unsqueeze(0).to(device)

                with torch.no_grad():
                    with autocast():
                        outputs = model(face_tensor)
                        _, predicted = torch.max(outputs, 1)
                        emotion = emotion_labels[predicted.item()]

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow('Facial Emotion Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# GUI with Tkinter
def create_gui():
    # Main window
    window = tk.Tk()
    window.title("Facial Emotion Recognition")
    window.geometry("400x300")

    # Label for model selection
    label = ttk.Label(window, text="Select the Model for Recognition:")
    label.pack(pady=20)

    # Dropdown for model selection
    selected_model = tk.StringVar()
    model_dropdown = ttk.Combobox(window, textvariable=selected_model)
    model_dropdown['values'] = ("ViTCN","DeiT", "CeiT", "CvT")
    model_dropdown.current(0)  # Default to first model
    model_dropdown.pack(pady=10)

    # Function to handle the start button click
    def on_start_click():
        model_name = selected_model.get()
        if model_name:
            start_recognition(model_name)
        else:
            messagebox.showerror("Error", "Please select a model.")

    # Start button to trigger emotion recognition
    start_button = ttk.Button(window, text="Start Recognition", command=on_start_click)
    start_button.pack(pady=80)

    # Run the GUI event loop
    window.mainloop()

# Run the GUI
if __name__ == "__main__":
    create_gui()