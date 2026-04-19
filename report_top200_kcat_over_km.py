#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Export top-200 rows by kcat/Km.")
    parser.add_argument("--input", type=str, default="blast500.csv", help="Input CSV path (default: blast500.csv).")
    parser.add_argument(
        "--top200-output",
        type=str,
        default="top200_kcat_over_km.csv",
        help="Top-200 output CSV path (default: top200_kcat_over_km.csv).",
    )
    return parser.parse_args()


def read_csv_auto(csv_path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030"]
    err = None
    for enc in encodings:
        try:
            return pd.read_csv(csv_path, encoding=enc)
        except Exception as e:
            err = e
    raise RuntimeError(f"Failed to read CSV: {csv_path}\nLast error: {err}")


def main():
    args = parse_args()
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    df = read_csv_auto(input_path).copy()
    if "Pred_kcat_over_Km" not in df.columns:
        raise ValueError("Required latest ratio column `Pred_kcat_over_Km` was not found in CSV.")
    df["kcat_over_Km"] = pd.to_numeric(df["Pred_kcat_over_Km"], errors="coerce")
    df = df.dropna(subset=["kcat_over_Km"]).reset_index(drop=True)

    # Sort all rows and assign 1-based rank.
    ranked = df.sort_values("kcat_over_Km", ascending=False, kind="mergesort").reset_index(drop=True)
    ranked["Rank"] = ranked.index + 1

    # Export top 200 rows.
    top200 = ranked.head(200).copy()
    top200_output = Path(args.top200_output).resolve()
    top200.to_csv(top200_output, index=False, encoding="utf-8-sig")

    print(f"Top-200 file saved: {top200_output}")
    print(top200.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

