#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _safe_pcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.size < 2:
        return float("nan")
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _safe_scc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.size < 2:
        return float("nan")
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))


@torch.no_grad()
def collect_predictions(model, dataloader, device):
    model.eval()
    all_y = []
    all_pred = []
    for batch in dataloader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        pred = model(x)
        all_y.append(y.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
    y_true = np.concatenate(all_y, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    return y_true, y_pred


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    loss_fn,
    device,
    max_grad_norm=None,
    train_noise_std: float = 0.0,
):
    model.train()
    total_loss = 0.0
    total_count = 0
    for batch in dataloader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        if train_noise_std is not None and train_noise_std > 0:
            # Add light input noise only during training to improve generalization.
            x = x + torch.randn_like(x) * float(train_noise_std)

        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_count += bs

    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device, target_inverse_transform=None) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0

    for batch in dataloader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_count += bs

    y_true, y_pred = collect_predictions(model, dataloader, device)
    if target_inverse_transform is not None:
        y_true_metric = target_inverse_transform(y_true)
        y_pred_metric = target_inverse_transform(y_pred)
    else:
        y_true_metric = y_true
        y_pred_metric = y_pred
    metrics = {
        "loss": total_loss / max(total_count, 1),
        "mse": float(mean_squared_error(y_true_metric, y_pred_metric)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_metric, y_pred_metric))),
        "mae": float(mean_absolute_error(y_true_metric, y_pred_metric)),
        "r2": float(r2_score(y_true_metric, y_pred_metric, multioutput="uniform_average")),
        "pcc": _safe_pcc(y_true_metric.reshape(-1), y_pred_metric.reshape(-1)),
        "scc": _safe_scc(y_true_metric.reshape(-1), y_pred_metric.reshape(-1)),
        "km_mse": float(mean_squared_error(y_true_metric[:, 0], y_pred_metric[:, 0])),
        "km_rmse": float(np.sqrt(mean_squared_error(y_true_metric[:, 0], y_pred_metric[:, 0]))),
        "km_mae": float(mean_absolute_error(y_true_metric[:, 0], y_pred_metric[:, 0])),
        "km_r2": float(r2_score(y_true_metric[:, 0], y_pred_metric[:, 0])),
        "km_pcc": _safe_pcc(y_true_metric[:, 0], y_pred_metric[:, 0]),
        "km_scc": _safe_scc(y_true_metric[:, 0], y_pred_metric[:, 0]),
        "kcat_mse": float(mean_squared_error(y_true_metric[:, 1], y_pred_metric[:, 1])),
        "kcat_rmse": float(np.sqrt(mean_squared_error(y_true_metric[:, 1], y_pred_metric[:, 1]))),
        "kcat_mae": float(mean_absolute_error(y_true_metric[:, 1], y_pred_metric[:, 1])),
        "kcat_r2": float(r2_score(y_true_metric[:, 1], y_pred_metric[:, 1])),
        "kcat_pcc": _safe_pcc(y_true_metric[:, 1], y_pred_metric[:, 1]),
        "kcat_scc": _safe_scc(y_true_metric[:, 1], y_pred_metric[:, 1]),
        "count": int(y_true_metric.shape[0]),
    }
    return metrics

