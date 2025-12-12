import torch
import torch.nn as nn


class BaseClassifier(nn.Module):
    """
    Base classifier for Task A. Simple feedforward network with interpretable layers.
    """

    def __init__(self, input_dim: int = 20, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 2)  # binary classification

    def forward(self, x):
        """Forward pass with hidden activations for interpretability."""
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        logits = self.fc3(h2)
        return logits, h2

    def get_probabilities(self, x):
        """Get prediction probabilities."""
        logits, _ = self.forward(x)
        return torch.softmax(logits, dim=1)
