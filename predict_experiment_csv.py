#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from config import CHECKPOINT_PATH, SCALER_PATH, TARGET_SCALER_PATH
from src.features.extractor import extract_joint_features
from src.models.multitask_model import MultiTaskRegressor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-predict Km and kcat from a CSV file."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="blast500.csv",
        help="Input CSV path (default: blast500.csv).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output CSV path (default: overwrite input file).",
    )
    parser.add_argument(
        "--seq-col",
        type=str,
        default="Enzyme",
        help="Protein sequence column name (default: Enzyme).",
    )
    parser.add_argument(
        "--smiles-col",
        type=str,
        default="Substrates",
        help="SMILES column name (default: Substrates).",
    )
    parser.add_argument(
        "--pred-kcat-col",
        type=str,
        default="Pred_kcat",
        help="Predicted kcat output column (default: Pred_kcat).",
    )
    parser.add_argument(
        "--pred-km-col",
        type=str,
        default="Pred_Km",
        help="Predicted Km output column (default: Pred_Km).",
    )
    parser.add_argument(
        "--pred-kcat-over-km-col",
        type=str,
        default="Pred_kcat_over_Km",
        help="Predicted kcat/Km output column (default: Pred_kcat_over_Km).",
    )
    parser.add_argument(
        "--pred-km-over-kcat-col",
        type=str,
        default="Pred_Km_over_kcat",
        help="Predicted Km/kcat output column (default: Pred_Km_over_kcat).",
    )
    return parser.parse_args()


def read_csv_auto(csv_path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(csv_path, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read CSV: {csv_path}\nLast error: {last_err}")


def build_model():
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
    return model, scaler, target_scaler


def main():
    args = parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    output_path = Path(args.output).resolve() if args.output.strip() else input_path
    df = read_csv_auto(input_path)

    if args.seq_col not in df.columns:
        raise ValueError(f"Protein sequence column not found: {args.seq_col}. Available columns: {list(df.columns)}")
    if args.smiles_col not in df.columns:
        raise ValueError(f"SMILES column not found: {args.smiles_col}. Available columns: {list(df.columns)}")

    sequences = df[args.seq_col].astype(str).str.strip().tolist()
    smiles_list = df[args.smiles_col].astype(str).str.strip().tolist()

    model, scaler, target_scaler = build_model()

    print(f"Start prediction, total samples: {len(df)}")
    features = extract_joint_features(smiles_list, sequences).astype(np.float32)
    features = scaler.transform(features).astype(np.float32)

    with torch.no_grad():
        pred_scaled = model(torch.tensor(features, dtype=torch.float32)).cpu().numpy().astype(np.float32)

    pred_log10 = target_scaler.inverse_transform(pred_scaled)
    km_log10 = pred_log10[:, 0]
    kcat_log10 = pred_log10[:, 1]

    km = np.power(10.0, km_log10)
    kcat = np.power(10.0, kcat_log10)

    # Compute ratios in log10 space, then convert back.
    km_over_kcat_log10 = km_log10 - kcat_log10
    km_over_kcat = np.power(10.0, km_over_kcat_log10)
    kcat_over_km = np.power(10.0, -km_over_kcat_log10)

    df[args.pred_kcat_col] = kcat
    df[args.pred_km_col] = km
    df[args.pred_km_over_kcat_col] = km_over_kcat
    df[args.pred_kcat_over_km_col] = kcat_over_km

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Prediction finished. Output written to: {output_path}")
    print(
        df[
            [
                args.pred_kcat_col,
                args.pred_km_col,
                args.pred_km_over_kcat_col,
                args.pred_kcat_over_km_col,
            ]
        ]
        .head(5)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()

