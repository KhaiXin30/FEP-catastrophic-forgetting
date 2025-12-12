import torch
import numpy as np
from typing import Dict

def compute_free_energy_from_logits(logits, targets):
    """
    Free energy for a batch: F(x) = -log p_true.
    logits:  (batch, num_classes)
    targets: (batch,)
    Returns:
        Tensor of shape (batch,)
    """
    probs = torch.softmax(logits, dim=1)
    probs = torch.clamp(probs, min=1e-10, max=1.0)
    idx = torch.arange(len(targets), device=logits.device)
    true_probs = probs[idx, targets]
    fe = -torch.log(true_probs)
    return fe

class FreeEnergyAnalyzer:
    """
    Measure & analyze free energy (surprise) for Task A vs Task B.
    """

    def __init__(self, model, X_tensor, y_task_a, y_task_b, device=None):
        self.model = model
        self.X_tensor = X_tensor
        self.y_task_a = torch.LongTensor(y_task_a)
        self.y_task_b = torch.LongTensor(y_task_b)
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model.to(self.device)

    def analyze_free_energy_landscape(self) -> Dict[str, np.ndarray]:
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(self.X_tensor.to(self.device))

        fe_task_a = compute_free_energy_from_logits(logits, self.y_task_a.to(self.device))
        fe_task_b = compute_free_energy_from_logits(logits, self.y_task_b.to(self.device))

        fe_divergence = (fe_task_b - fe_task_a).mean().item()
        fe_uncertainty_a = fe_task_a.std().item()
        fe_uncertainty_b = fe_task_b.std().item()

        return {
            "mean_fe_task_a": fe_task_a.mean().item(),
            "mean_fe_task_b": fe_task_b.mean().item(),
            "fe_divergence": fe_divergence,
            "fe_uncertainty_a": fe_uncertainty_a,
            "fe_uncertainty_b": fe_uncertainty_b,
            "fe_task_a": fe_task_a.cpu().numpy(),
            "fe_task_b": fe_task_b.cpu().numpy(),
        }

    @staticmethod
    def compute_alignment_score(results: Dict[str, float]) -> float:
        """
        Map FE divergence to [0,1] alignment score via sigmoid:
        1.0 -> strongly Task-A aligned
        0.0 -> strongly Task-B aligned
        ~0.5 -> neutral.
        """
        divergence = results["fe_divergence"]
        return 1.0 / (1.0 + np.exp(-divergence * 5.0))
