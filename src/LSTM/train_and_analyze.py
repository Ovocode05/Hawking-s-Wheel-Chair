import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from dataset import WheelchairDataset
from model import LSTMFeatureExtractor

import sys

# -----------------
# Configuration
# -----------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_ROOT = os.path.join(PROJECT_ROOT, "Normalized_dataset", "recordings")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "src", "LSTM", "results")
LOG_FILE = os.path.join(PROJECT_ROOT, "src", "LSTM", "training_run.log")

# Redirect stdout/stderr to log file
class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()

sys.stdout = Tee(LOG_FILE, 'w')


BATCH_SIZE = 16
HIDDEN_SIZE = 64
NUM_LAYERS = 2
LEARNING_RATE = 0.001
EPOCHS = 100
SPLIT_RATIO = 0.8

os.makedirs(OUTPUT_DIR, exist_ok=True)

def train_model():
    print("--- Starting Training Pipeline ---")
    
    # 1. Dataset
    dataset = WheelchairDataset(data_root=DATA_ROOT, sequence_length=150)
    
    if len(dataset) == 0:
        print("Error: No data found. Exiting.")
        return
    
    # Split
    train_size = int(SPLIT_RATIO * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Data Split: {len(train_dataset)} Train, {len(val_dataset)} Validation")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = LSTMFeatureExtractor(
        input_size=5, 
        hidden_size=HIDDEN_SIZE, 
        num_layers=NUM_LAYERS, 
        num_classes=len(dataset.label_to_index)
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 3. Training Loop
    train_losses = []
    val_losses = []
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_loss)
        
        # Validation
        model.eval()
        val_running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_epoch_loss = val_running_loss / len(val_dataset)
        val_losses.append(val_epoch_loss)
        val_acc = 100 * correct / total
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
    # 4. Save Model
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "lstm_model.pth"))
    
    # 5. Plot Loss
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
    plt.close()
    
    # 6. Feature Analysis (Using ALL data to see full distribution)
    analyze_features(model, dataset, device)

def analyze_features(model, dataset, device):
    print("--- Starting Feature Analysis ---")
    model.eval()
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels, _ in loader:
            inputs = inputs.to(device)
            features = model.forward_features(inputs)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
            
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # t-SNE
    print("Running t-SNE...")
    # Perplexity should be smaller than number of points. Default 30.
    # If dataset is small, reduce perplexity.
    n_samples = all_features.shape[0]
    perplexity = min(30, n_samples - 1) if n_samples > 1 else 1
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    features_2d = tsne.fit_transform(all_features)
    
    # Plot
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(all_labels)
    
    for label_idx in unique_labels:
        # Get class name
        class_name = dataset.index_to_label[label_idx]
        indices = all_labels == label_idx
        plt.scatter(features_2d[indices, 0], features_2d[indices, 1], label=class_name, alpha=0.7)
        
    plt.title('t-SNE Visualization of LSTM Features')
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_clusters.png"))
    plt.close()
    
    print(f"Analysis plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    train_model()
