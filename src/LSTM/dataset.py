import os
import glob
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class WheelchairDataset(Dataset):
    def __init__(self, data_root, sequence_length=150):
        """
        Args:
            data_root (str): Path to the root of the dataset (e.g., .../Normalized_dataset/recordings)
            sequence_length (int): Expected length of the sequence. 
                                   If data is shorter, it will be padded (or ignored). 
                                   If longer, truncated.
        """
        self.data_root = data_root
        self.sequence_length = sequence_length
        self.files = []
        self.labels = []
        self.label_to_index = {}
        self.index_to_label = {}
        
        self._scan_dataset()

    def _scan_dataset(self):
        print(f"Scanning dataset at {self.data_root}...")
        
        # Get all class directories
        classes = [d for d in os.listdir(self.data_root) if os.path.isdir(os.path.join(self.data_root, d))]
        classes.sort()
        
        self.label_to_index = {cls_name: idx for idx, cls_name in enumerate(classes)}
        self.index_to_label = {idx: cls_name for idx, cls_name in enumerate(classes)}
        
        print(f"Found classes: {classes}")
        
        for cls_name in classes:
            cls_dir = os.path.join(self.data_root, cls_name)
            csv_files = glob.glob(os.path.join(cls_dir, "*.csv"))
            
            for file_path in csv_files:
                # Basic check if file is valid/not empty (can do more in __getitem__)
                if os.path.getsize(file_path) > 0:
                    self.files.append(file_path)
                    self.labels.append(self.label_to_index[cls_name])
                else:
                    print(f"Warning: Skipping empty file {file_path}")
                    
        print(f"Total valid files found: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]
        
        try:
            # Load CSV
            df = pd.read_csv(file_path)
            
            # Extract features.
            # User requested: theta, omega, alpha, x_dis, y_dis.
            # CSV contains: frame_idx,t,theta,x,y,omega,alpha
            # We map x_dis -> x, y_dis -> y
            required_cols = ['theta', 'omega', 'alpha', 'x', 'y']
            
            # check if all columns exist
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                print(f"Error: Missing columns {missing_cols} in {file_path}")
                raise ValueError(f"Missing columns: {missing_cols}")

            # Convert to numpy
            # Shape (Seq, 5)
            features = df[required_cols].values.astype(np.float32)
            
            # Handle Length
            seq_len = len(features)
            
            if seq_len < self.sequence_length:
                # Pad (seq_len, 5) -> (150, 5)
                # We pad the first dimension (time), keep 2nd dim (features) same
                pad_width_time = self.sequence_length - seq_len
                # pad_width: ((top, bottom), (left, right))
                # We want ((0, pad), (0, 0))
                features = np.pad(features, ((0, pad_width_time), (0, 0)), mode='constant', constant_values=0)
            elif seq_len > self.sequence_length:
                # Truncate
                features = features[:self.sequence_length, :]
            
            # Reshape is not needed for (Seq, Feat) if we extracted correctly
            # But sanity check shape
            # features shape is (150, 5)
            
            # Convert to Tensor
            features_tensor = torch.from_numpy(features)
            label_tensor = torch.tensor(label, dtype=torch.long)
            
            return features_tensor, label_tensor, file_path

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return zero-tensor as fallback
            return torch.zeros((self.sequence_length, 5)), torch.tensor(label, dtype=torch.long), "ERROR"
            
