#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json

import joblib
import numpy as np
import torch

from config import CHECKPOINT_PATH, SCALER_PATH, TARGET_SCALER_PATH
from src.features.extractor import extract_joint_features
from src.models.multitask_model import MultiTaskRegressor


def parse_args():
    parser = argparse.ArgumentParser(description="Run unified model inference for Km and kcat.")
    parser.add_argument("--protein", type=str, required=True, help="Protein sequence.")
    parser.add_argument("--smiles", type=str, required=True, help="Substrate SMILES.")
    parser.add_argument("--device", type=str, default="cpu", help="Inference device (default: cpu).")
    return parser.parse_args()


def main():
    args = parse_args()

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {CHECKPOINT_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Feature scaler not found: {SCALER_PATH}")
    if not TARGET_SCALER_PATH.exists():
        raise FileNotFoundError(f"Target scaler not found: {TARGET_SCALER_PATH}")

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    scaler = joblib.load(SCALER_PATH)
    target_scaler = joblib.load(TARGET_SCALER_PATH)

    model = MultiTaskRegressor(
        input_dim=ckpt["input_dim"],
        hidden_dim=ckpt["hidden_dim"],
        dropout=ckpt["dropout"],
        use_mha=ckpt.get("use_mha", True),
        attn_tokens=ckpt.get("attn_tokens", 4),
        attn_dim=ckpt.get("attn_dim", 64),
        attn_heads=ckpt.get("attn_heads", 4),
        use_gate=ckpt.get("use_gate", True),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    device = torch.device(args.device)
    model.to(device)

    features = extract_joint_features([args.smiles], [args.protein]).astype(np.float32)
    features = scaler.transform(features).astype(np.float32)

    x = torch.tensor(features, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = model(x)
    pred = pred.cpu().numpy().astype(np.float32)
    pred = target_scaler.inverse_transform(pred)

    km_log10 = float(pred[0, 0])
    kcat_log10 = float(pred[0, 1])

    result = {
        "input": {
            "protein": args.protein[:80] + ("..." if len(args.protein) > 80 else ""),
            "smiles": args.smiles,
        },
        "prediction": {
            "log10_km": km_log10,
            "km": float(10 ** km_log10),
            "log10_kcat": kcat_log10,
            "kcat": float(10 ** kcat_log10),
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

