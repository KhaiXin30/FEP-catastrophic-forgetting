import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def set_global_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SyntheticDataGenerator:
    """
    Generates synthetic data for two related but distinct tasks:

    - Task A: depends on features 0-4
    - Task B: depends on features 5-9
    """

    def __init__(self, n_samples: int = 1000, n_features: int = 20, seed: int = 42):
        self.n_samples = n_samples
        self.n_features = n_features
        set_global_seed(seed)

    def generate_task_a_data(self):
        """
        Generate (X, y_A, weights_A)

        X: (n_samples, n_features)
        y_A: binary labels based on features 0-4
        """
        X = np.random.randn(self.n_samples, self.n_features)

        weights_a = np.zeros(self.n_features)
        weights_a[:5] = np.array([0.5, 0.4, 0.3, 0.2, 0.1])

        logits_a = X @ weights_a + np.random.randn(self.n_samples) * 0.1
        y_a = (logits_a > 0).astype(int)

        return X, y_a, weights_a

    def generate_task_b_data(self, X):
        """
        Generate (y_B, weights_B) using the SAME X as Task A,
        but based on DIFFERENT features 5–9.
        """
        weights_b = np.zeros(self.n_features)
        weights_b[5:10] = np.array([0.5, 0.4, 0.3, 0.2, 0.1])

        logits_b = X @ weights_b + np.random.randn(self.n_samples) * 0.1
        y_b = (logits_b > 0).astype(int)

        return y_b, weights_b

    def prepare_loaders(self, X, y, batch_size: int = 32, test_split: float = 0.2):
        """
        Create train/test dataloaders for a single task.
        """
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)

        n_test = int(len(X) * test_split)
        indices = np.random.permutation(len(X))
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        train_loader = DataLoader(
            TensorDataset(X_tensor[train_idx], y_tensor[train_idx]),
            batch_size=batch_size,
            shuffle=True,
        )

        test_loader = DataLoader(
            TensorDataset(X_tensor[test_idx], y_tensor[test_idx]),
            batch_size=batch_size,
            shuffle=False,
        )

        return train_loader, test_loader, X_tensor, y_tensor

    def prepare_dual_task_loaders(
        self,
        X,
        y_task_a,
        y_task_b,
        batch_size: int = 32,
        test_split: float = 0.2,
    ):
        """
        Create loaders that return (X, y_A, y_B) together.
        Useful if you want both labels in the same batch.
        """
        X_tensor = torch.FloatTensor(X)
        y_a_tensor = torch.LongTensor(y_task_a)
        y_b_tensor = torch.LongTensor(y_task_b)

        n_test = int(len(X) * test_split)
        indices = np.random.permutation(len(X))
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        train_loader = DataLoader(
            TensorDataset(
                X_tensor[train_idx], y_a_tensor[train_idx], y_b_tensor[train_idx]
            ),
            batch_size=batch_size,
            shuffle=True,
        )

        test_loader = DataLoader(
            TensorDataset(
                X_tensor[test_idx], y_a_tensor[test_idx], y_b_tensor[test_idx]
            ),
            batch_size=batch_size,
            shuffle=False,
        )

        return train_loader, test_loader, X_tensor, y_a_tensor, y_b_tensor
