#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys

from config import UNIFIED_DATASET_PATH
from src.data.build_dataset import build_unified_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Entry point for the Km+kcat multitask deep-learning project.")
    sub = parser.add_subparsers(dest="command")

    build_parser = sub.add_parser("build-dataset", help="Validate and clean the dataset CSV.")
    build_parser.add_argument("--input", type=str, default=str(UNIFIED_DATASET_PATH), help="Input CSV path.")

    sub.add_parser("train", help="Train the multitask model (calls train.py).")
    predict_parser = sub.add_parser("predict", help="Predict Km and kcat (calls predict.py).")
    predict_parser.add_argument("--protein", type=str, required=True, help="Protein sequence.")
    predict_parser.add_argument("--smiles", type=str, required=True, help="Substrate SMILES.")
    predict_parser.add_argument("--device", type=str, default="cpu", help="Inference device.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "build-dataset":
        build_unified_dataset(args.input)
        return 0

    if args.command == "train":
        cmd = f"{sys.executable} train.py"
        return os.system(cmd)

    if args.command == "predict":
        cmd = (
            f"{sys.executable} predict.py "
            f"--protein \"{args.protein}\" "
            f"--smiles \"{args.smiles}\" "
            f"--device {args.device}"
        )
        return os.system(cmd)

    print("Available commands: build-dataset / train / predict")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
