import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTModel
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# Define the custom dataset class
class FER2013Dataset(Dataset):
    def __init__(self, csv_file, usage, transform=None):
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data['Usage'] == usage]
        self.data.reset_index(drop=True, inplace=True)  # Reset index
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = np.fromstring(self.data.loc[idx, 'pixels'], sep=' ').reshape(48, 48).astype(np.uint8)
        emotion = int(self.data.loc[idx, 'emotion'])

        # Convert grayscale image to RGB by duplicating the single channel three times
        image = np.stack([image] * 3, axis=-1)

        if self.transform:
            image = self.transform(image)

        return image, emotion

# Data transformation and augmentation
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Resize to 224x224 for ViT
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

val_test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Resize to 224x224 for ViT
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Load datasets
train_dataset = FER2013Dataset(csv_file='/kaggle/input/fer2013/fer2013.csv', usage='Training', transform=train_transform)
val_dataset = FER2013Dataset(csv_file='/kaggle/input/fer2013/fer2013.csv', usage='PublicTest', transform=val_test_transform)
test_dataset = FER2013Dataset(csv_file='/kaggle/input/fer2013/fer2013.csv', usage='PrivateTest', transform=val_test_transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

# Define the model architecture
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

# Initialize the model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ViTCN(num_classes=7).to(device)
model.load_state_dict(torch.load('/kaggle/input/vitcn-model/final_model.pth'))
model.eval()

# LIME setup
explainer = lime_image.LimeImageExplainer()

# Function to make predictions
def predict(images):
    # Ensure images are numpy arrays of uint8
    images = [(img * 255).astype(np.uint8) for img in images]
    
    # Preprocess images to match the model input format
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # Resize to 224x224 for ViT
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    images = [preprocess(img) for img in images]
    images = torch.stack(images).to(device)
    
    with torch.no_grad():
        outputs = model(images)
    probs = torch.nn.functional.softmax(outputs, dim=1)
    return probs.cpu().numpy()

# Function to visualize LIME explanation for an image
def explain_image(img):
    explanation = explainer.explain_instance(
        img,
        predict,
        top_labels=5,
        hide_color=0,
        num_samples=1000
    )

    # Get the explanation for the top label
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=10,
        hide_rest=False
    )

    # Create a copy of the original image to apply the mask
    img_with_boundary = img.copy()

    # Apply the mask to highlight the explanation with a black boundary
    boundary_image = mark_boundaries(temp / 255.0, mask)

    return boundary_image

# Mapping of label numbers to emotion names
label_to_emotion = {
    0: "Anger",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

# Function to display images with LIME explanations
def display_images_with_lime(dataset, label, num_images=2):
    # Filter dataset for the specified label
    label_indices = [i for i, row in dataset.data.iterrows() if row['emotion'] == label]
    
    # Randomly select the specified number of images
    sampled_indices = np.random.choice(label_indices, num_images, replace=False)
    
    fig, axs = plt.subplots(num_images, 2, figsize=(10, 5 * num_images))
    for i, idx in enumerate(sampled_indices):
        image, _ = dataset[idx]
        image = image.permute(1, 2, 0).numpy()
        original_image = (image * 0.5 + 0.5) * 255  # Unnormalize image to [0, 255]
        
        lime_img = explain_image(original_image.astype(np.uint8))

        # Plot original image
        axs[i, 0].imshow(original_image.astype(np.uint8))
        axs[i, 0].set_title(f"Original: {label_to_emotion[label]}")
        axs[i, 0].axis('off')
        
        # Plot LIME-explained image
        axs[i, 1].imshow(lime_img)
        axs[i, 1].set_title(f"LIME: {label_to_emotion[label]}")
        axs[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

# Display 2 random images with LIME explanations for each label
for label in range(7):
    display_images_with_lime(test_dataset, label, num_images=2)