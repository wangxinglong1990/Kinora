#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
from pathlib import Path

import pandas as pd

from config import DATA_DIR, UNIFIED_DATASET_PATH


def _find_col(df: pd.DataFrame, candidates):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        real = lower_map.get(cand.lower())
        if real is not None:
            return real
    return None


def normalize_km_kcat_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    sequence_col = _find_col(df, ["Sequence", "sequence"])
    smiles_col = _find_col(df, ["Smiles", "smiles"])
    km_col = _find_col(df, ["Km(M)", "km(m)", "Km", "km"])
    kcat_col = _find_col(df, ["kcat(s^-1)", "kcat", "Kcat"])
    fold_col = _find_col(df, ["fold", "Fold"])

    missing = []
    if sequence_col is None:
        missing.append("Sequence")
    if smiles_col is None:
        missing.append("Smiles/smiles")
    if km_col is None:
        missing.append("Km(M)")
    if kcat_col is None:
        missing.append("kcat(s^-1)")
    if missing:
        raise ValueError(f"Dataset is missing required fields: {', '.join(missing)}")

    selected_cols = [sequence_col, smiles_col, km_col, kcat_col]
    renamed_cols = ["Sequence", "smiles", "km", "kcat"]
    if fold_col is not None:
        selected_cols.append(fold_col)
        renamed_cols.append("fold")

    clean_df = df[selected_cols].copy()
    clean_df.columns = renamed_cols

    clean_df = clean_df.dropna(subset=["Sequence", "smiles", "km", "kcat"])
    clean_df["Sequence"] = clean_df["Sequence"].astype(str).str.strip()
    clean_df["smiles"] = clean_df["smiles"].astype(str).str.strip()
    clean_df["km"] = pd.to_numeric(clean_df["km"], errors="coerce")
    clean_df["kcat"] = pd.to_numeric(clean_df["kcat"], errors="coerce")
    clean_df = clean_df.dropna(subset=["km", "kcat"])
    clean_df = clean_df[(clean_df["Sequence"].str.len() > 0) & (clean_df["smiles"].str.len() > 0)]
    clean_df = clean_df[(clean_df["km"] > 0) & (clean_df["kcat"] > 0)]
    if "fold" in clean_df.columns:
        clean_df["fold"] = pd.to_numeric(clean_df["fold"], errors="coerce")
        clean_df = clean_df.dropna(subset=["fold"])
        clean_df["fold"] = clean_df["fold"].astype(int)

    clean_df["km_log10"] = clean_df["km"].apply(math.log10)
    clean_df["kcat_log10"] = clean_df["kcat"].apply(math.log10)
    output_cols = ["Sequence", "smiles", "km", "kcat", "km_log10", "kcat_log10"]
    if "fold" in clean_df.columns:
        output_cols.append("fold")
    return clean_df[output_cols].reset_index(drop=True)


def assess_log10_suitability(df: pd.DataFrame, col: str):
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    s = s[s > 0]
    if len(s) == 0:
        return {
            "column": col,
            "recommended": False,
            "reason": "No valid positive samples.",
        }

    min_v = float(s.min())
    max_v = float(s.max())
    p1 = float(s.quantile(0.01))
    p99 = float(s.quantile(0.99))
    orders = float(math.log10(max_v / min_v)) if min_v > 0 else float("inf")
    skew_raw = float(s.skew())
    skew_log = float(s.apply(math.log10).skew())
    recommended = (orders >= 2.0) or (abs(skew_raw) > 1.0 and abs(skew_log) < abs(skew_raw))

    return {
        "column": col,
        "count": int(len(s)),
        "min": min_v,
        "p01": p1,
        "p99": p99,
        "max": max_v,
        "orders_of_magnitude": orders,
        "skew_raw": skew_raw,
        "skew_log10": skew_log,
        "recommended": bool(recommended),
    }


def build_unified_dataset(input_path=UNIFIED_DATASET_PATH, output_path=None, show_report=True):
    dataset_path = Path(input_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    raw_df = pd.read_csv(dataset_path, low_memory=False)
    clean_df = normalize_km_kcat_dataframe(raw_df)
    km_report = assess_log10_suitability(clean_df, "km")
    kcat_report = assess_log10_suitability(clean_df, "kcat")

    if output_path is not None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        clean_df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"Cleaned unified dataset saved: {output_path}")

    if show_report:
        print(f"Total samples: {len(clean_df)}")
        print(f"log10 suitability report (km): {km_report}")
        print(f"log10 suitability report (kcat): {kcat_report}")
    return clean_df

