import torch
import torch.nn as nn

class LSTMFeatureExtractor(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, num_classes=5, dropout=0.3):
        """
        Args:
            input_size (int): Number of input features per time step (default 5).
            hidden_size (int): Size of the LSTM hidden state.
            num_layers (int): Number of LSTM layers.
            num_classes (int): Number of output classes (words).
            dropout (float): Dropout probability.
        """
        super(LSTMFeatureExtractor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Layer
        # batch_first=True means input shape is (batch, seq, feature)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Classifier Head (for supervision)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward_features(self, x):
        """
        Extracts features (embeddings) from the input sequence.
        Returns the last hidden state of the LSTM.
        """
        # x shape: (batch_size, seq_len, input_size)
        
        # Reset hidden state is automatic if not provided, which works for disjoint batches
        # out shape: (batch_size, seq_len, hidden_size)
        # h_n shape: (num_layers, batch_size, hidden_size)
        out, (h_n, c_n) = self.lstm(x)
        
        # We take the output of the last time step
        # out[:, -1, :] shape: (batch_size, hidden_size)
        feature_vector = out[:, -1, :]
        
        return feature_vector

    def forward(self, x):
        """
        Standard forward pass for training.
        Returns class logits.
        """
        # Get features
        features = self.forward_features(x)
        
        # Pass through classifier
        logits = self.classifier(features)
        
        return logits
