#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from config import (
    CHECKPOINT_PATH,
    METRICS_PATH,
    MODELS_DIR,
    SCALER_PATH,
    TARGET_SCALER_PATH,
    UNIFIED_DATASET_PATH,
)
from src.data.build_dataset import build_unified_dataset
from src.data.dataset import MultiTaskKineticsDataset
from src.features.extractor import extract_joint_features
from src.losses import TaskWeightedMSELoss
from src.models.multitask_model import MultiTaskRegressor
from src.trainer import collect_predictions, evaluate, train_one_epoch
from src.visualization.paper_figures import generate_paper_figures


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the multitask deep model (Km + kcat).")
    parser.add_argument("--rebuild-dataset", action="store_true", help="Backward-compatible flag; dataset cleaning already runs by default.")
    parser.add_argument("--dataset", type=str, default=str(UNIFIED_DATASET_PATH), help="Input dataset CSV path.")
    parser.add_argument("--show-data-audit", action="store_true", help="Show sample counts and log10 fitness reports.")
    parser.add_argument("--limit", type=int, default=0, help="Debug option: cap sample size; 0 means all rows.")
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Lower bound of learning rate.")
    parser.add_argument("--lr-factor", type=float, default=0.5, help="Learning-rate decay factor.")
    parser.add_argument("--lr-patience", type=int, default=2, help="Number of non-improving test evaluations before LR decay.")
    parser.add_argument("--hidden-dim", type=int, default=192, help="Shared hidden dimension.")
    parser.add_argument("--dropout", type=float, default=0.45, help="Dropout ratio.")
    parser.add_argument("--weight-decay", type=float, default=0.03, help="AdamW weight decay.")
    parser.add_argument("--train-noise-std", type=float, default=0.01, help="Input noise std during training; 0 disables it.")
    parser.add_argument("--use-mha", dest="use_mha", action="store_true", help="Enable lightweight multi-head attention.")
    parser.add_argument("--no-mha", dest="use_mha", action="store_false", help="Disable multi-head attention.")
    parser.set_defaults(use_mha=True)
    parser.add_argument("--attn-tokens", type=int, default=4, help="Number of attention tokens.")
    parser.add_argument("--attn-dim", type=int, default=64, help="Attention token dimension.")
    parser.add_argument("--attn-heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--use-gate", dest="use_gate", action="store_true", help="Enable gated attention fusion.")
    parser.add_argument("--no-gate", dest="use_gate", action="store_false", help="Disable gated attention fusion.")
    parser.set_defaults(use_gate=True)
    parser.add_argument("--loss-type", type=str, default="huber", choices=["huber", "mse"], help="Regression loss type.")
    parser.add_argument("--huber-beta", type=float, default=0.6, help="Huber loss beta.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping threshold; <=0 disables clipping.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-5, help="Minimum test-loss improvement for early-stop reset.")
    parser.add_argument("--test-every", type=int, default=5, help="Run test evaluation every N epochs.")
    parser.add_argument("--test-patience", type=int, default=10, help="Early stop after N non-improving test evaluations.")
    parser.add_argument("--km-weight", type=float, default=1.0, help="Loss weight for Km task.")
    parser.add_argument("--kcat-weight", type=float, default=1.0, help="Loss weight for kcat task.")
    parser.add_argument("--device", type=str, default="auto", help="cpu/cuda/auto")
    return parser.parse_args()


def resolve_device(device_arg: str):
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_float(v):
    if isinstance(v, (np.floating, np.integer)):
        v = float(v)
    if isinstance(v, float) and np.isnan(v):
        return None
    return float(v)


def _sanitize_metrics(d: dict):
    return {k: _to_float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v for k, v in d.items()}


def _aggregate_fold_metrics(fold_metrics):
    if not fold_metrics:
        return {}
    keys = [k for k in fold_metrics[0].keys() if k != "fold"]
    summary = {}
    for k in keys:
        vals = np.array([m[k] for m in fold_metrics], dtype=np.float64)
        valid = vals[~np.isnan(vals)]
        if valid.size == 0:
            summary[f"{k}_mean"] = None
            summary[f"{k}_std"] = None
        else:
            summary[f"{k}_mean"] = float(np.mean(valid))
            summary[f"{k}_std"] = float(np.std(valid, ddof=0))
    return summary


def main():
    args = parse_args()
    if args.test_every <= 0:
        raise ValueError("--test-every must be greater than 0.")
    if args.test_patience <= 0:
        raise ValueError("--test-patience must be greater than 0.")
    if args.lr_patience <= 0:
        raise ValueError("--lr-patience must be greater than 0.")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay cannot be negative.")
    if args.train_noise_std < 0:
        raise ValueError("--train-noise-std cannot be negative.")
    set_seed(args.seed)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(args.dataset)
    df = build_unified_dataset(input_path=dataset_path, show_report=args.show_data_audit)
    if args.limit and args.limit > 0:
        df = df.sample(n=min(args.limit, len(df)), random_state=args.seed).reset_index(drop=True)
    if "fold" not in df.columns:
        raise ValueError("Missing `fold` column in dataset; cannot run k-fold cross-validation.")

    sequences = df["Sequence"].astype(str).tolist()
    smiles = df["smiles"].astype(str).tolist()
    targets = df[["km_log10", "kcat_log10"]].values.astype(np.float32)
    fold_ids = df["fold"].astype(int).values

    print(f"Start feature extraction, total samples: {len(df)}")
    all_features = extract_joint_features(smiles, sequences).astype(np.float32)

    device = resolve_device(args.device)
    print(f"Training device: {device}")

    unique_folds = sorted(pd.unique(fold_ids))
    if len(unique_folds) != 10:
        print(f"Warning: detected {len(unique_folds)} folds instead of 10. Cross-validation will use detected folds.")

    history_rows = []
    fold_test_metrics = []
    prediction_rows = []
    first_fold_scaler = None
    first_fold_target_scaler = None
    first_fold_ckpt = None

    for fold in unique_folds:
        test_mask = fold_ids == int(fold)
        trainval_mask = ~test_mask

        x_train = all_features[trainval_mask]
        y_train = targets[trainval_mask]
        x_test = all_features[test_mask]
        y_test = targets[test_mask]

        feature_scaler = StandardScaler()
        x_train = feature_scaler.fit_transform(x_train).astype(np.float32)
        x_test = feature_scaler.transform(x_test).astype(np.float32)

        target_scaler = StandardScaler()
        y_train_scaled = target_scaler.fit_transform(y_train).astype(np.float32)
        y_test_scaled = target_scaler.transform(y_test).astype(np.float32)

        train_ds = MultiTaskKineticsDataset(x_train, y_train_scaled)
        test_ds = MultiTaskKineticsDataset(x_test, y_test_scaled)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        model = MultiTaskRegressor(
            input_dim=x_train.shape[1],
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            use_mha=args.use_mha,
            attn_tokens=args.attn_tokens,
            attn_dim=args.attn_dim,
            attn_heads=args.attn_heads,
            use_gate=args.use_gate,
        ).to(device)
        loss_fn = TaskWeightedMSELoss(
            km_weight=args.km_weight,
            kcat_weight=args.kcat_weight,
            loss_type=args.loss_type,
            huber_beta=args.huber_beta,
        )
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        effective_lr_patience = args.lr_patience
        if effective_lr_patience >= args.test_patience:
            effective_lr_patience = max(1, args.test_patience - 1)
            print(
                f"Fold {fold} adjusted lr_patience: original={args.lr_patience} >= test_patience={args.test_patience}, "
                f"set to {effective_lr_patience} so LR decay can occur before early stopping."
            )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_factor,
            patience=effective_lr_patience,
            threshold=args.early_stop_min_delta,
            threshold_mode="abs",
            min_lr=args.min_lr,
        )

        best_test_loss = float("inf")
        best_state = None
        best_epoch = 0
        bad_test_evals = 0
        best_train_metrics = None
        best_test_metrics = None

        print(f"========== Fold {fold} / {len(unique_folds)} ==========")
        for epoch in range(1, args.epochs + 1):
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                loss_fn,
                device,
                max_grad_norm=args.max_grad_norm,
                train_noise_std=args.train_noise_std,
            )
            train_metrics = evaluate(
                model,
                train_loader,
                loss_fn,
                device,
                target_inverse_transform=target_scaler.inverse_transform,
            )
            run_test_eval = (epoch % args.test_every == 0)
            test_metrics_now = None
            improved = False
            lr_reduced = False
            if run_test_eval:
                test_metrics_now = evaluate(
                    model,
                    test_loader,
                    loss_fn,
                    device,
                    target_inverse_transform=target_scaler.inverse_transform,
                )
                prev_lr = float(optimizer.param_groups[0]["lr"])
                scheduler.step(test_metrics_now["loss"])
                current_lr_after_step = float(optimizer.param_groups[0]["lr"])
                lr_reduced = current_lr_after_step < (prev_lr - 1e-15)
                if test_metrics_now["loss"] < (best_test_loss - args.early_stop_min_delta):
                    best_test_loss = test_metrics_now["loss"]
                    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                    best_epoch = epoch
                    best_train_metrics = dict(train_metrics)
                    best_test_metrics = dict(test_metrics_now)
                    bad_test_evals = 0
                    improved = True
                else:
                    bad_test_evals += 1
            current_lr = float(optimizer.param_groups[0]["lr"])

            history_rows.append(
                {
                    "fold": int(fold),
                    "epoch": int(epoch),
                    "train_loss": train_metrics["loss"],
                    "train_pcc": train_metrics["pcc"],
                    "train_rmse": train_metrics["rmse"],
                    "train_km_pcc": train_metrics["km_pcc"],
                    "train_km_rmse": train_metrics["km_rmse"],
                    "train_kcat_pcc": train_metrics["kcat_pcc"],
                    "train_kcat_rmse": train_metrics["kcat_rmse"],
                    "lr": current_lr,
                    "test_eval": int(run_test_eval),
                    "test_loss": None if test_metrics_now is None else test_metrics_now["loss"],
                    "test_km_pcc": None if test_metrics_now is None else test_metrics_now["km_pcc"],
                    "test_km_rmse": None if test_metrics_now is None else test_metrics_now["km_rmse"],
                    "test_kcat_pcc": None if test_metrics_now is None else test_metrics_now["kcat_pcc"],
                    "test_kcat_rmse": None if test_metrics_now is None else test_metrics_now["kcat_rmse"],
                    "bad_test_evals": bad_test_evals,
                    "lr_reduced": int(lr_reduced),
                }
            )

            if test_metrics_now is None:
                print(
                    f"Fold {fold} Epoch {epoch}/{args.epochs} | "
                    f"train_loss={train_metrics['loss']:.4f} train_pcc={train_metrics['pcc']:.4f} train_rmse={train_metrics['rmse']:.4f} | "
                    f"lr={current_lr:.2e}"
                )
            else:
                print(
                    f"Fold {fold} Epoch {epoch}/{args.epochs} | "
                    f"train_loss={train_metrics['loss']:.4f} train_pcc={train_metrics['pcc']:.4f} train_rmse={train_metrics['rmse']:.4f} | "
                    f"TEST: km_pcc={test_metrics_now['km_pcc']:.4f} km_rmse={test_metrics_now['km_rmse']:.4f}, "
                    f"kcat_pcc={test_metrics_now['kcat_pcc']:.4f} kcat_rmse={test_metrics_now['kcat_rmse']:.4f} | "
                    f"improved={int(improved)} bad_test_evals={bad_test_evals} lr_drop={int(lr_reduced)} lr={current_lr:.2e}"
                )
                if bad_test_evals >= args.test_patience:
                    print(
                        f"Fold {fold} early stop triggered: {args.test_patience} consecutive non-improving test evaluations, "
                        f"stopped at epoch {epoch}."
                    )
                    break

        if best_state is None:
            fallback_test = evaluate(
                model,
                test_loader,
                loss_fn,
                device,
                target_inverse_transform=target_scaler.inverse_transform,
            )
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_epoch = args.epochs
            best_train_metrics = dict(train_metrics)
            best_test_metrics = dict(fallback_test)

        model.load_state_dict(best_state)
        test_metrics = best_test_metrics if best_test_metrics is not None else evaluate(
            model,
            test_loader,
            loss_fn,
            device,
            target_inverse_transform=target_scaler.inverse_transform,
        )
        y_true_test, y_pred_test = collect_predictions(model, test_loader, device)
        y_true_test = target_scaler.inverse_transform(y_true_test)
        y_pred_test = target_scaler.inverse_transform(y_pred_test)
        for idx in range(y_true_test.shape[0]):
            true_km_log10 = float(y_true_test[idx, 0])
            true_kcat_log10 = float(y_true_test[idx, 1])
            pred_km_log10 = float(y_pred_test[idx, 0])
            pred_kcat_log10 = float(y_pred_test[idx, 1])
            prediction_rows.append(
                {
                    "fold": int(fold),
                    "true_km_log10": true_km_log10,
                    "pred_km_log10": pred_km_log10,
                    "true_kcat_log10": true_kcat_log10,
                    "pred_kcat_log10": pred_kcat_log10,
                    "true_eff_log10": true_kcat_log10 - true_km_log10,
                    "pred_eff_log10": pred_kcat_log10 - pred_km_log10,
                }
            )
        test_metrics["fold"] = int(fold)
        test_metrics["best_epoch"] = int(best_epoch)
        if best_train_metrics is not None:
            test_metrics["best_train_loss"] = float(best_train_metrics["loss"])
            test_metrics["best_train_km_pcc"] = float(best_train_metrics["km_pcc"])
            test_metrics["best_train_kcat_pcc"] = float(best_train_metrics["kcat_pcc"])
        fold_test_metrics.append(_sanitize_metrics(test_metrics))

        print(
            f"Fold {fold} TEST | "
            f"KM: pcc={test_metrics['km_pcc']:.4f} scc={test_metrics['km_scc']:.4f} rmse={test_metrics['km_rmse']:.4f} | "
            f"KCAT: pcc={test_metrics['kcat_pcc']:.4f} scc={test_metrics['kcat_scc']:.4f} rmse={test_metrics['kcat_rmse']:.4f}"
        )

        if first_fold_scaler is None:
            first_fold_scaler = feature_scaler
            first_fold_target_scaler = target_scaler
            first_fold_ckpt = {
                "state_dict": model.state_dict(),
                "input_dim": int(x_train.shape[1]),
                "hidden_dim": int(args.hidden_dim),
                "dropout": float(args.dropout),
                "use_mha": bool(args.use_mha),
                "attn_tokens": int(args.attn_tokens),
                "attn_dim": int(args.attn_dim),
                "attn_heads": int(args.attn_heads),
                "use_gate": bool(args.use_gate),
            }
            # Save first-fold model/scalers immediately for safety.
            joblib.dump(first_fold_scaler, SCALER_PATH)
            joblib.dump(first_fold_target_scaler, TARGET_SCALER_PATH)
            torch.save(first_fold_ckpt, CHECKPOINT_PATH)
            print("=== First-fold model checkpoint and scalers saved locally ===")

        # Release large objects no longer needed.
        del model, optimizer, train_loader, test_loader, x_train, y_train, x_test, y_test
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Final save step: write aggregated CV artifacts.
    history_path = MODELS_DIR / "cv_history.csv"
    pd.DataFrame(history_rows).to_csv(history_path, index=False, encoding="utf-8")

    fold_metrics_path = MODELS_DIR / "cv_fold_test_metrics.csv"
    pd.DataFrame(fold_test_metrics).to_csv(fold_metrics_path, index=False, encoding="utf-8")
    prediction_path = MODELS_DIR / "cv_test_predictions.csv"
    pred_df = pd.DataFrame(prediction_rows)
    pred_df.to_csv(prediction_path, index=False, encoding="utf-8")

    summary_metrics = _aggregate_fold_metrics(fold_test_metrics)
    summary_metrics = {k: v for k, v in summary_metrics.items() if not k.startswith("eff_")}
    final_report = {
        "cv_type": "10-fold_by_dataset_fold_column",
        "num_folds": int(len(unique_folds)),
        "folds": [int(f) for f in unique_folds],
        "per_fold_test_metrics": [{k: v for k, v in m.items() if not str(k).startswith("eff_")} for m in fold_test_metrics],
        "cv_test_summary": summary_metrics,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)

    generate_paper_figures(
        data_df=df,
        pred_df=pred_df,
        output_dir=MODELS_DIR / "figures",
    )

    print(f"Training/validation history saved: {history_path}")
    print(f"Per-fold test metrics saved: {fold_metrics_path}")
    print(f"Cross-validation test predictions saved: {prediction_path}")
    print(f"Cross-validation summary saved: {METRICS_PATH}")
    print(f"Paper-style figures (a,b,c) saved: {MODELS_DIR / 'figures'}")
    print(json.dumps(final_report["cv_test_summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

