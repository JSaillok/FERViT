import sys
# Add the path to the CEIT repository to the path
sys.path.append('/kaggle/input/ceit-model/ceit-main')

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from ceit_model import ceit_small_patch16_224
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

# Initialize the model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ceit_small_patch16_224(pretrained=True).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)  # L2 regularization added

# Early stopping and learning rate scheduler
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.best_model_wts = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_wts = model.state_dict()
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                model.load_state_dict(self.best_model_wts)
        else:
            self.best_loss = val_loss
            self.best_model_wts = model.state_dict()
            self.counter = 0

# Training function with early stopping 
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20, patience=7):
    early_stopping = EarlyStopping(patience=patience)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, min_lr=1e-6)
    
    best_loss = float('inf')
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Training]'):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(train_accuracy)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {train_accuracy:.4f}')

        # Validate the model
        val_loss = 0.0
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Validation]'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print('Early stopping')
            break
        
        # Learning rate scheduling
        scheduler.step(val_loss)

    # Save the final model
    torch.save(model.state_dict(), 'final_model.pth')
    print('Model saved as final_model.pth')
    
    # Plot losses and accuracies
    epochs_range = range(1, epoch + 2)
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')

    plt.show()
    
# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer)

# Load the best model
model.load_state_dict(torch.load('final_model.pth'))

# Class label mapping
id2label = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}

# Evaluate the model on the test set
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')

    # Confusion matrix and classification report
    cm = confusion_matrix(all_labels, all_predictions, labels=[0, 1, 2, 3, 4, 5, 6])

    # Normalize confusion matrix by rows (i.e by the number of samples in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=list(id2label.values()))
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Normalized)')
    plt.show()

    # Classification Report
    report = classification_report(all_labels, all_predictions, target_names=list(id2label.values()), zero_division=0)
    print("Classification Report:")
    print(report)

# Call the evaluation function
evaluate_model(model, test_loader, device)