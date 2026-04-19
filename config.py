#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"
SMILES_TRANSFORMER_DIR = PROJECT_DIR / "SMILES_Transform"
SMILES_TRANSFORMER_CHECKPOINT = SMILES_TRANSFORMER_DIR / "trfm_12_23000.pkl"
PROTEIN_ESMC_WEIGHTS_PATH = DATA_DIR / "weights" / "esmc_300m_2024_12_v0.pth"
PROTEIN_ESMC_MODEL_NAME = "esmc_300m"

UNIFIED_DATASET_PATH = DATA_DIR / "kcat-over-Km-data_0.4simi-10fold.csv"

CHECKPOINT_PATH = MODELS_DIR / "multitask_dl.pt"
METRICS_PATH = MODELS_DIR / "metrics.json"
SCALER_PATH = MODELS_DIR / "feature_scaler.joblib"
TARGET_SCALER_PATH = MODELS_DIR / "target_scaler.joblib"

