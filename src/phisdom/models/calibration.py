from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim


class TemperatureScaler(nn.Module):
    """One-parameter temperature scaling for binary logits.

    Fits T > 0 on validation set to minimize BCEWithLogits loss on scaled logits (logits / T).
    """

    def __init__(self, init_T: float = 1.0):
        super().__init__()
        self.log_T = nn.Parameter(torch.tensor(float(init_T)).log())

    @property
    def T(self) -> torch.Tensor:
        return torch.exp(self.log_T)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.T.clamp(min=1e-3, max=100.0)


@dataclass
class TempScaleResult:
    T: float
    val_loss_before: float
    val_loss_after: float


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 200, lr: float = 0.01) -> Tuple[TemperatureScaler, TempScaleResult]:
    device = logits.device
    ts = TemperatureScaler().to(device)
    criterion = nn.BCEWithLogitsLoss()
    opt = optim.Adam([ts.log_T], lr=lr)
    with torch.no_grad():
        val_loss_before = float(criterion(logits, labels.float()).item())
    for _ in range(max_iter):
        opt.zero_grad(set_to_none=True)
        scaled = ts(logits)
        loss = criterion(scaled, labels.float())
        loss.backward()
        opt.step()
    with torch.no_grad():
        val_loss_after = float(criterion(ts(logits), labels.float()).item())
    return ts, TempScaleResult(T=float(ts.T.detach().cpu().item()), val_loss_before=val_loss_before, val_loss_after=val_loss_after)
