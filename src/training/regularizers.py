
from copy import deepcopy
from typing import Tuple, List

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict

from ..data import set_global_seed  # if you have this; otherwise remove
from ..models.lora import LoRAClassifier
from ..fe.metrics import compute_free_energy_from_logits



# ------------------------------------------------------------------
# Helper: choose device
# ------------------------------------------------------------------
def _get_device(device: str = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------
# Helper: dual-task dataloaders (Task A + Task B labels)
# ------------------------------------------------------------------
def prepare_dual_task_loaders(
    X,
    y_task_a,
    y_task_b,
    batch_size: int = 32,
    test_split: float = 0.2,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train / test DataLoaders for the dual-task setting.

    Each batch will yield (X_batch, y_b_batch, y_a_batch) so that:
      - y_b is the main objective (Task B)
      - y_a is available for FE regularization (Task A).
    """
    X = np.asarray(X)
    y_task_a = np.asarray(y_task_a)
    y_task_b = np.asarray(y_task_b)

    n_samples = X.shape[0]
    n_test = int(n_samples * test_split)
    indices = np.random.permutation(n_samples)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    X_train = torch.FloatTensor(X[train_idx])
    y_a_train = torch.LongTensor(y_task_a[train_idx])
    y_b_train = torch.LongTensor(y_task_b[train_idx])

    X_test = torch.FloatTensor(X[test_idx])
    y_a_test = torch.LongTensor(y_task_a[test_idx])
    y_b_test = torch.LongTensor(y_task_b[test_idx])

    # IMPORTANT: order = (X, y_b, y_a) to match your original training loop
    train_ds = TensorDataset(X_train, y_b_train, y_a_train)
    test_ds = TensorDataset(X_test, y_b_test, y_a_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# ------------------------------------------------------------------
# Main routine: original LoRA variant training + strength sweep
# ------------------------------------------------------------------
# def train_lora_variant(
#     base_model: nn.Module,
#     X,
#     y_task_a,
#     y_task_b,
#     regularizer: str = "none",  # "none", "l2", "l1", "fe"
#     lambda_reg: float = 0.1,
#     r: int = 4,
#     lr: float = 1e-3,
#     epochs: int = 50,
#     batch_size: int = 32,
#     test_split: float = 0.2,
#     device: str = None,
#     seed: int | None = None,
# ) -> Tuple[nn.Module, pd.DataFrame]:
#     """
#     Original behaviour:

#     - Freeze base model (Task A solution).
#     - Wrap it with LoRA for the final FC layer.
#     - Train **only** the LoRA weights on Task B.
#     - During training, fix `lora_strength = 1.0` (full influence).
#     - Regularizers:
#         * "none": CE on Task B only
#         * "l2":   weight_decay on LoRA params (Adam)
#         * "l1":   L1 penalty on LoRA params (+ scalar)
#         * "fe":   Free energy on Task A labels (per-example -log p(y_A))
#     - After training, sweep LoRA strength in {0, 0.2, ..., 1.0}
#       and compute:
#         * acc_a, acc_b
#         * fe_a, fe_b
#         * fe_div = fe_b - fe_a

#     Returns
#     -------
#     lora_model : nn.Module
#         Trained LoRA-wrapped model.
#     results_df : pd.DataFrame
#         One row per LoRA strength with all metrics.
#     """
#     dev = _get_device(device)

#     if seed is not None:
#         random.seed(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)

#     # 1. Copy base model and wrap with LoRA
#     base_copy = deepcopy(base_model).to(dev)
#     lora_model = LoRAClassifier(base_copy, r=r).to(dev)

#     # 2. Freeze base model weights (keep Task A solution fixed)
#     for p in lora_model.base_model.parameters():
#         p.requires_grad = False

#     # 3. Fix LoRA strength **during training** so adapter gets full gradient
#     lora_model.lora_strength.data = torch.tensor(1.0, device=dev)
#     lora_model.lora_strength.requires_grad = False  # scalar is not trained

#     # 4. Only train the LoRA weights of the final layer
#     lora_params: List[torch.nn.Parameter] = list(lora_model.lora_fc3.parameters())

#     # 5. Optimizer
#     if regularizer == "l2":
#         optimizer = torch.optim.Adam(lora_params, lr=lr, weight_decay=lambda_reg)
#     else:
#         optimizer = torch.optim.Adam(lora_params, lr=lr)

#     criterion = nn.CrossEntropyLoss()

#     # 6. Data loaders (dual-task)
#     train_loader, _ = prepare_dual_task_loaders(
#         X, y_task_a, y_task_b, batch_size=batch_size, test_split=test_split
#     )

#     # 7. Training loop
#     for epoch in range(epochs):
#         lora_model.train()
#         total_loss = 0.0

#         for X_batch, y_b_batch, y_a_batch in train_loader:
#             X_batch = X_batch.to(dev)
#             y_b_batch = y_b_batch.to(dev)
#             y_a_batch = y_a_batch.to(dev)

#             optimizer.zero_grad()
#             logits, _ = lora_model(X_batch)

#             # Main loss: Task B classification
#             loss = criterion(logits, y_b_batch)

#             # Probabilities for FE
#             probs = torch.softmax(logits, dim=1)
#             probs = torch.clamp(probs, min=1e-10, max=1.0)

#             # Regularizers
#             if regularizer == "fe":
#                 # Free energy on Task A labels: -log p(y_A)
#                 fe_a = -torch.log(
#                     probs[torch.arange(len(X_batch), device=dev), y_a_batch]
#                 )
#                 loss = loss + lambda_reg * fe_a.mean()

#             elif regularizer == "l1":
#                 l1_penalty = torch.tensor(0.0, device=dev)
#                 for p in lora_model.lora_fc3.parameters():
#                     l1_penalty = l1_penalty + p.abs().sum()
#                 # Optionally include the scalar strength
#                 l1_penalty = l1_penalty + lora_model.lora_strength.abs()
#                 loss = loss + lambda_reg * l1_penalty

#             # "none" and "l2" handled implicitly (L2 via weight_decay)

#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()

#         if (epoch + 1) % 10 == 0:
#             avg_loss = total_loss / len(train_loader)
#             print(
#                 f"[{regularizer.upper()}] Epoch {epoch+1}/{epochs} | "
#                 f"Loss: {avg_loss:.4f}"
#             )

#     # 8. Evaluate across LoRA strengths on the **full** dataset
#     X_tensor = torch.FloatTensor(np.asarray(X)).to(dev)
#     y_a_tensor = torch.LongTensor(np.asarray(y_task_a)).to(dev)
#     y_b_tensor = torch.LongTensor(np.asarray(y_task_b)).to(dev)

#     strengths = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
#     rows = []

#     lora_model.eval()
#     with torch.no_grad():
#         for s in strengths:
#             # set LoRA strength
#             lora_model.lora_strength.data = torch.tensor(float(s), device=dev)

#             logits, _ = lora_model(X_tensor)
#             preds = torch.argmax(logits, dim=1)

#             acc_a = (preds == y_a_tensor).float().mean().item()
#             acc_b = (preds == y_b_tensor).float().mean().item()

#             fe_a = compute_free_energy_from_logits(logits, y_a_tensor).mean().item()
#             fe_b = compute_free_energy_from_logits(logits, y_b_tensor).mean().item()
#             fe_div = fe_b - fe_a

#             rows.append(
#                 {
#                     "acc_a": acc_a,
#                     "acc_b": acc_b,
#                     "fe_a": fe_a,
#                     "fe_b": fe_b,
#                     "fe_div": fe_div,
#                     "regularizer": regularizer,
#                     "lambda": lambda_reg,
#                     "strength": s,
#                 }
#             )

#     results_df = pd.DataFrame(rows)
#     return lora_model, results_df

def evaluate_model_metrics(model,
                           X_tensor,
                           y_a_tensor,
                           y_b_tensor,
                           device) -> Dict[str, float]:
    """
    Evaluate Task A/B accuracy and FE_A / FE_B for a given model
    at its current LoRA strength.
    """
    model.eval()
    X_tensor = X_tensor.to(device)
    y_a_tensor = y_a_tensor.to(device)
    y_b_tensor = y_b_tensor.to(device)

    with torch.no_grad():
        logits, _ = model(X_tensor)
        preds = torch.argmax(logits, dim=1)

        acc_a = (preds == y_a_tensor).float().mean().item()
        acc_b = (preds == y_b_tensor).float().mean().item()

        fe_a = compute_free_energy_from_logits(logits, y_a_tensor)
        fe_b = compute_free_energy_from_logits(logits, y_b_tensor)

    fe_a_mean = fe_a.mean().item()
    fe_b_mean = fe_b.mean().item()
    fe_divergence = fe_b_mean - fe_a_mean

    return {
        "acc_a": acc_a,
        "acc_b": acc_b,
        "fe_a": fe_a_mean,
        "fe_b": fe_b_mean,
        "fe_div": fe_divergence,
    }



def train_lora_variant(
    base_model: nn.Module,
    X: np.ndarray,
    y_task_a: np.ndarray,
    y_task_b: np.ndarray,
    regularizer: str = "none",   # "none", "l1", "l2", "fe"
    lambda_reg: float = 0.1,
    r: int = 4,
    lr: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 32,
    test_split: float = 0.2,
    device: torch.device = None,
) -> Tuple[nn.Module, pd.DataFrame]:

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fresh copy of base model
    base_copy = deepcopy(base_model).to(device)

    # Wrap with LoRA
    lora_model = LoRAClassifier(base_copy, r=r).to(device)

    # Freeze base model
    for p in lora_model.base_model.parameters():
        p.requires_grad = False

    # During training, use full LoRA strength and keep scalar fixed
    lora_model.lora_strength.data = torch.tensor(1.0, device=device)
    lora_model.lora_strength.requires_grad = False

    # Only train the LoRA A/B weights
    lora_params = list(lora_model.lora_fc3.parameters())

    # Optimizer (L2 via weight_decay if chosen)
    if regularizer == "l2":
        optimizer = torch.optim.Adam(lora_params, lr=lr, weight_decay=lambda_reg)
    else:
        optimizer = torch.optim.Adam(lora_params, lr=lr)

    criterion = nn.CrossEntropyLoss()

    # Dual-task loaders so batches include y_B and y_A
    train_loader, _ = prepare_dual_task_loaders(
        X, y_task_a, y_task_b, batch_size=batch_size, test_split=test_split
    )

    # ---- Training loop ----
    for epoch in range(epochs):
        lora_model.train()
        total_loss = 0.0

        for X_batch, y_b_batch, y_a_batch in train_loader:
            X_batch = X_batch.to(device)
            y_b_batch = y_b_batch.to(device)
            y_a_batch = y_a_batch.to(device)

            optimizer.zero_grad()
            logits, _ = lora_model(X_batch)

            probs = torch.softmax(logits, dim=1)
            probs = torch.clamp(probs, min=1e-10, max=1.0)

            loss = criterion(logits, y_b_batch)  # main Task-B CE

            # ---- Regularizers ----
            if regularizer == "fe":
                # FE penalty on Task A labels
                idx = torch.arange(len(X_batch), device=device)
                true_probs = probs[idx, y_a_batch]
                fe_a = -torch.log(true_probs)
                loss = loss + lambda_reg * fe_a.mean()

            elif regularizer == "l1":
                l1_penalty = 0.0
                for p in lora_model.lora_fc3.parameters():
                    l1_penalty = l1_penalty + p.abs().sum()
                l1_penalty = l1_penalty + lora_model.lora_strength.abs()
                loss = loss + lambda_reg * l1_penalty
            # "none" and "l2" are already handled

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"[{regularizer.upper()}] Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    # ---- Evaluation across LoRA strengths ----
    X_tensor_full = torch.FloatTensor(X)
    y_a_tensor_full = torch.LongTensor(y_task_a)
    y_b_tensor_full = torch.LongTensor(y_task_b)

    strengths = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    results = []

    for s in strengths:
        lora_model.lora_strength.data = torch.tensor(float(s), device=device)
        metrics = evaluate_model_metrics(
            lora_model, X_tensor_full, y_a_tensor_full, y_b_tensor_full, device
        )
        metrics.update({
            "regularizer": regularizer,
            "lambda": lambda_reg,
            "strength": s,
        })
        results.append(metrics)

    return lora_model, pd.DataFrame(results)