import torch
import torch.nn as nn
import torch.optim as optim


class ModelTrainer:
    """
    Generic trainer for a classifier with (logits, _) output.
    """

    def __init__(
        self,
        model,
        learning_rate: float = 1e-3,
        device: str = "cpu",
        trainable_params=None,
    ):
        self.model = model.to(device)
        self.device = device

        if trainable_params is None:
            trainable_params = self.model.parameters()

        self.optimizer = optim.Adam(trainable_params, lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.history = {"train_loss": [], "test_accuracy": []}

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            logits, _ = self.model(X)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def evaluate(self, data_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in data_loader:
                X, y = X.to(self.device), y.to(self.device)
                logits, _ = self.model(X)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / total

    def train_model(self, train_loader, test_loader, epochs: int = 50, verbose: bool = True):
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            test_acc = self.evaluate(test_loader)
            self.history["train_loss"].append(train_loss)
            self.history["test_accuracy"].append(test_acc)

            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Loss: {train_loss:.4f} | Accuracy: {test_acc:.4f}"
                )
        return self.history
