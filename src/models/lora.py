import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer: W * x + Bx where B=BA*BB with rank r.
    """

    def __init__(self, in_features: int, out_features: int, r: int = 4):
        super().__init__()
        self.r = r
        self.lora_a = nn.Linear(in_features, r, bias=False)
        self.lora_b = nn.Linear(r, out_features, bias=False)

        nn.init.normal_(self.lora_a.weight, std=0.02)
        nn.init.normal_(self.lora_b.weight, std=0.02)

    def forward(self, x):
        # (batch, in_features) -> (batch, out_features)
        return self.lora_b(self.lora_a(x))


class LoRAClassifier(nn.Module):
    """
    Wraps a BaseClassifier and adds a LoRA layer on top of the final hidden layer.
    """

    def __init__(self, base_model: nn.Module, r: int = 4):
        super().__init__()
        self.base_model = base_model
        hidden_dim_half = base_model.hidden_dim // 2
        self.lora_fc3 = LoRALayer(hidden_dim_half, 2, r=r)
        # Mixing scalar that controls LoRA strength
        self.lora_strength = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        # Reuse base model layers but add LoRA to final logits
        h1 = self.base_model.relu(self.base_model.fc1(x))
        h2 = self.base_model.relu(self.base_model.fc2(h1))
        logits_base = self.base_model.fc3(h2)
        logits_lora = self.lora_fc3(h2)

        logits = logits_base + self.lora_strength * logits_lora
        return logits, h2

    def get_probabilities(self, x):
        logits, _ = self.forward(x)
        return torch.softmax(logits, dim=1)
