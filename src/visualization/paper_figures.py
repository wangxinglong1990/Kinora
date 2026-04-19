#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

FIG_DPI = 600


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def generate_paper_figures(data_df: pd.DataFrame, pred_df: pd.DataFrame, output_dir: Path):
    _ensure_dir(output_dir)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.titlesize": 24,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "figure.titlesize": 24,
        }
    )
    sns.set_theme(style="whitegrid", font_scale=1.5)

    # a) Distribution of kcat/Km values
    eff_raw = data_df["kcat"] / data_df["km"]
    eff_log10 = np.log10(eff_raw.values.astype(np.float64))
    plt.figure(figsize=(7, 5))
    sns.histplot(eff_log10, bins=50, kde=True, color="#4C72B0")
    plt.xlabel("log10(kcat/Km)")
    plt.ylabel("Count")
    plt.title("")
    plt.tight_layout()
    plt.savefig(output_dir / "fig_a_kcat_over_km_distribution.png", dpi=FIG_DPI)
    plt.close()

    # b) Violin plot by fold with median line
    if "fold" in data_df.columns:
        fold_df = data_df.copy()
        fold_df["eff_log10"] = np.log10(fold_df["kcat"] / fold_df["km"])
        plt.figure(figsize=(10, 5))
        ax = sns.violinplot(
            data=fold_df,
            x="fold",
            y="eff_log10",
            palette="tab10",
            inner=None,
            cut=0,
            linewidth=1,
        )
        medians = fold_df.groupby("fold")["eff_log10"].median().sort_index()
        for i, fold in enumerate(medians.index):
            ax.plot([i - 0.25, i + 0.25], [medians.loc[fold], medians.loc[fold]], color="white", lw=2.2)
        plt.xlabel("Fold")
        plt.ylabel("log10(kcat/Km)")
        plt.title("")
        plt.tight_layout()
        plt.savefig(output_dir / "fig_b_fold_violin.png", dpi=FIG_DPI)
        plt.close()

    # c) Performance scatter with density colorbar (hexbin)
    plt.figure(figsize=(6, 6))
    hb = plt.hexbin(
        pred_df["true_eff_log10"].values,
        pred_df["pred_eff_log10"].values,
        gridsize=55,
        cmap="viridis",
        mincnt=1,
    )
    vmin = float(min(pred_df["true_eff_log10"].min(), pred_df["pred_eff_log10"].min()))
    vmax = float(max(pred_df["true_eff_log10"].max(), pred_df["pred_eff_log10"].max()))
    plt.plot([vmin, vmax], [vmin, vmax], "--", color="red", lw=1.2)
    cbar = plt.colorbar(hb)
    cbar.set_label("Data density")
    plt.xlabel("Experimental log10(kcat/Km)")
    plt.ylabel("Predicted log10(kcat/Km)")
    plt.title("")
    plt.tight_layout()
    plt.savefig(output_dir / "fig_c_density_scatter.png", dpi=FIG_DPI)
    plt.close()

    # d/e/f comparison bar charts are intentionally skipped during training.

