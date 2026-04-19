#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

def parse_args():
    parser = argparse.ArgumentParser(
        description="Cluster protein sequences from top200_kcat_over_km.csv and generate a 600-dpi UMAP figure."
    )
    parser.add_argument("--input", type=str, default="top200_kcat_over_km.csv", help="Input CSV (default: top200_kcat_over_km.csv).")
    parser.add_argument("--output-csv", type=str, default="clustered_top200_kcat_over_km.csv", help="Output clustered CSV path.")
    parser.add_argument("--output-fig", type=str, default="protein_cluster_umap_top200_600dpi.png", help="Output cluster figure (PNG).")
    parser.add_argument("--output-metrics", type=str, default="clustering_metrics_top200.csv", help="Output clustering metrics CSV.")
    parser.add_argument("--seq-col", type=str, default="Enzyme", help="Protein sequence column name.")
    parser.add_argument("--min-cluster-size", type=int, default=8, help="HDBSCAN min_cluster_size (>=8 recommended).")
    parser.add_argument("--min-samples", type=int, default=3, help="HDBSCAN min_samples (typically < min_cluster_size).")
    parser.add_argument("--label-guided-umap", action="store_true", help="Enable label-guided UMAP for clearer visualization.")
    parser.add_argument("--no-label-guided-umap", dest="label_guided_umap", action="store_false", help="Disable label-guided UMAP.")
    parser.set_defaults(label_guided_umap=True)
    parser.add_argument("--auto-umap-tune", action="store_true", help="Auto-search UMAP parameters for better separation.")
    parser.add_argument("--no-auto-umap-tune", dest="auto_umap_tune", action="store_false", help="Disable UMAP auto-tuning.")
    parser.set_defaults(auto_umap_tune=True)
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--keep-noise",
        action="store_true",
        help="Keep HDBSCAN noise label (-1). By default, noise points are reassigned to nearest clusters.",
    )
    return parser.parse_args()


def read_csv_auto(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read CSV: {path}\nLast error: {last_err}")


def get_protein_embeddings(sequences):
    """
    Use the project ESMC encoder (same path as training/prediction).
    """
    from src.features.extractor import _get_protein_encoder

    encoder = _get_protein_encoder()
    emb = encoder.encode(sequences).astype(np.float32)
    return emb


def run_hdbscan_or_fallback(embeddings, min_cluster_size=8, min_samples=3, seed=42):
    """
    Use HDBSCAN when available; otherwise fallback to agglomerative clustering.
    """
    labels = None
    method = "HDBSCAN"

    try:
        import hdbscan

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(2, int(min_cluster_size)),
            min_samples=max(1, int(min_samples)),
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(embeddings)
    except Exception:
        method = "Agglomerative(fallback)"
        best_score = -1.0
        best_labels = None
        n = embeddings.shape[0]
        # For small sets, search k in [2, 8].
        for k in range(2, min(8, n - 1) + 1):
            model = AgglomerativeClustering(n_clusters=k, metric="euclidean", linkage="ward")
            cand = model.fit_predict(embeddings)
            if len(np.unique(cand)) < 2:
                continue
            score = silhouette_score(embeddings, cand, metric="euclidean")
            if score > best_score:
                best_score = score
                best_labels = cand
        if best_labels is None:
            best_labels = np.zeros(n, dtype=int)
        labels = best_labels

    return labels, method


def _layout_separation_score(xy, labels):
    """
    Score 2D layout separation using:
    - silhouette on non-noise points
    - inter-cluster centroid distance / intra-cluster radius
    """
    labels = np.asarray(labels)
    xy = np.asarray(xy, dtype=np.float64)
    keep = labels != -1
    if np.sum(keep) < 10:
        return -1e9
    labs = labels[keep]
    pts = xy[keep]
    uniq = np.unique(labs)
    if len(uniq) < 2:
        return -1e9
    sil = silhouette_score(pts, labs, metric="euclidean")

    centroids = []
    radii = []
    for c in uniq:
        p = pts[labs == c]
        mu = p.mean(axis=0)
        centroids.append(mu)
        radii.append(np.linalg.norm(p - mu, axis=1).mean() + 1e-8)
    centroids = np.asarray(centroids)
    radii = np.asarray(radii)
    # Minimum distance between cluster centroids.
    min_centroid_dist = np.inf
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            d = np.linalg.norm(centroids[i] - centroids[j])
            if d < min_centroid_dist:
                min_centroid_dist = d
    compact = float(np.median(radii))
    sep_ratio = float(min_centroid_dist / max(compact, 1e-8))
    # Primary objective: silhouette; secondary: centroid separation ratio.
    return float(sil + 0.15 * np.log1p(max(sep_ratio, 0.0)))


def run_umap(embeddings, labels=None, seed=42, auto_tune=True, label_guided=True):
    """
    Run UMAP for 2D visualization.
    """
    try:
        import umap
    except Exception as e:
        raise ImportError(
            "Missing dependency `umap-learn`. Install it first: pip install umap-learn"
        ) from e

    if not auto_tune:
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=14,
            min_dist=0.12,
            spread=2.0,
            metric="cosine",
            random_state=seed,
        )
        if label_guided and labels is not None and len(np.unique(labels[labels != -1])) >= 2:
            y = np.where(labels == -1, np.max(labels) + 1, labels)
            return reducer.fit_transform(embeddings, y=y)
        return reducer.fit_transform(embeddings)

    # Auto-search parameters for better visual separation.
    n_neighbors_grid = [8, 12, 18, 26]
    min_dist_grid = [0.02, 0.08, 0.16, 0.28]
    spread_grid = [1.5, 2.2, 3.0]

    best_xy = None
    best_cfg = None
    best_score = -1e9

    has_valid_labels = labels is not None and len(np.unique(labels[labels != -1])) >= 2
    for nn in n_neighbors_grid:
        for md in min_dist_grid:
            for sp in spread_grid:
                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=nn,
                    min_dist=md,
                    spread=sp,
                    metric="cosine",
                    random_state=seed,
                )
                try:
                    if label_guided and has_valid_labels:
                        y = np.where(labels == -1, np.max(labels) + 1, labels)
                        xy = reducer.fit_transform(embeddings, y=y)
                    else:
                        xy = reducer.fit_transform(embeddings)
                    score = _layout_separation_score(xy, labels if labels is not None else np.zeros(len(xy), dtype=int))
                except Exception:
                    continue
                if score > best_score:
                    best_score = score
                    best_xy = xy
                    best_cfg = (nn, md, sp)

    if best_xy is None:
        raise RuntimeError("UMAP parameter search failed. Please check input data or environment.")
    print(f"Best UMAP params: n_neighbors={best_cfg[0]}, min_dist={best_cfg[1]}, spread={best_cfg[2]}, score={best_score:.4f}")
    return best_xy


def compute_clustering_metrics(embeddings, umap_xy, labels):
    labels = np.asarray(labels, dtype=int)
    n_total = int(len(labels))
    noise_mask = labels == -1
    n_noise = int(np.sum(noise_mask))
    noise_ratio = float(n_noise / max(n_total, 1))

    non_noise_mask = ~noise_mask
    y_nn = labels[non_noise_mask]
    X_emb_nn = embeddings[non_noise_mask]
    X_umap_nn = umap_xy[non_noise_mask]
    unique_clusters = np.unique(y_nn)
    n_clusters = int(len(unique_clusters))

    metrics = {
        "n_total": n_total,
        "n_noise": n_noise,
        "noise_ratio": noise_ratio,
        "n_clusters_excluding_noise": n_clusters,
    }

    # Cluster size distribution metrics.
    if n_clusters > 0:
        cluster_sizes = np.array([(y_nn == c).sum() for c in unique_clusters], dtype=np.int64)
        p = cluster_sizes.astype(np.float64) / np.sum(cluster_sizes)
        shannon_h = float(-np.sum(p * np.log(p + 1e-12)))
        shannon_h_norm = float(shannon_h / math.log(max(n_clusters, 2)))
        effective_cluster_n = float(np.exp(shannon_h))
        hhi = float(np.sum(p ** 2))
        metrics.update(
            {
                "largest_cluster_size": int(cluster_sizes.max()),
                "smallest_cluster_size": int(cluster_sizes.min()),
                "largest_to_smallest_ratio": float(cluster_sizes.max() / max(cluster_sizes.min(), 1)),
                "shannon_entropy": shannon_h,
                "shannon_entropy_normalized": shannon_h_norm,
                "effective_number_of_clusters": effective_cluster_n,
                "herfindahl_index": hhi,
            }
        )
    else:
        metrics.update(
            {
                "largest_cluster_size": np.nan,
                "smallest_cluster_size": np.nan,
                "largest_to_smallest_ratio": np.nan,
                "shannon_entropy": np.nan,
                "shannon_entropy_normalized": np.nan,
                "effective_number_of_clusters": np.nan,
                "herfindahl_index": np.nan,
            }
        )

    # Geometric clustering metrics (requires at least 2 clusters).
    if n_clusters >= 2 and X_umap_nn.shape[0] > n_clusters:
        metrics["silhouette_umap"] = float(silhouette_score(X_umap_nn, y_nn, metric="euclidean"))
        metrics["davies_bouldin_umap"] = float(davies_bouldin_score(X_umap_nn, y_nn))
        metrics["calinski_harabasz_umap"] = float(calinski_harabasz_score(X_umap_nn, y_nn))
    else:
        metrics["silhouette_umap"] = np.nan
        metrics["davies_bouldin_umap"] = np.nan
        metrics["calinski_harabasz_umap"] = np.nan

    # Also report metrics in original embedding space for reference.
    if n_clusters >= 2 and X_emb_nn.shape[0] > n_clusters:
        metrics["silhouette_embedding"] = float(silhouette_score(X_emb_nn, y_nn, metric="euclidean"))
        metrics["davies_bouldin_embedding"] = float(davies_bouldin_score(X_emb_nn, y_nn))
        metrics["calinski_harabasz_embedding"] = float(calinski_harabasz_score(X_emb_nn, y_nn))
    else:
        metrics["silhouette_embedding"] = np.nan
        metrics["davies_bouldin_embedding"] = np.nan
        metrics["calinski_harabasz_embedding"] = np.nan

    # Extra UMAP-space compactness/separation metrics.
    if n_clusters >= 2:
        centers = []
        radii = []
        for c in unique_clusters:
            pts = X_umap_nn[y_nn == c]
            mu = pts.mean(axis=0)
            centers.append(mu)
            radii.append(float(np.linalg.norm(pts - mu, axis=1).mean()))
        centers = np.asarray(centers, dtype=np.float64)
        radii = np.asarray(radii, dtype=np.float64)

        min_center_dist = np.inf
        max_center_dist = 0.0
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                d = float(np.linalg.norm(centers[i] - centers[j]))
                min_center_dist = min(min_center_dist, d)
                max_center_dist = max(max_center_dist, d)
        sep_ratio = float(min_center_dist / max(np.median(radii), 1e-12))
        metrics["mean_cluster_radius_umap"] = float(np.mean(radii))
        metrics["median_cluster_radius_umap"] = float(np.median(radii))
        metrics["min_center_distance_umap"] = float(min_center_dist)
        metrics["max_center_distance_umap"] = float(max_center_dist)
        metrics["separation_ratio_umap"] = sep_ratio
    else:
        metrics["mean_cluster_radius_umap"] = np.nan
        metrics["median_cluster_radius_umap"] = np.nan
        metrics["min_center_distance_umap"] = np.nan
        metrics["max_center_distance_umap"] = np.nan
        metrics["separation_ratio_umap"] = np.nan

    return metrics


def assign_noise_to_nearest_cluster(embeddings, labels):
    """
    Reassign noise points (-1) to nearest cluster centroids.
    Returns: (new_labels, reassigned_noise_count)
    """
    labels = np.asarray(labels, dtype=int).copy()
    noise_mask = labels == -1
    n_noise = int(np.sum(noise_mask))
    if n_noise == 0:
        return labels, 0

    valid_clusters = sorted([int(c) for c in np.unique(labels) if c != -1])
    # Edge case: if all are noise, assign all to cluster 0.
    if len(valid_clusters) == 0:
        labels[:] = 0
        return labels, n_noise

    centroids = np.vstack([embeddings[labels == c].mean(axis=0) for c in valid_clusters])
    noise_idx = np.where(noise_mask)[0]
    noise_emb = embeddings[noise_idx]
    dist = np.linalg.norm(noise_emb[:, None, :] - centroids[None, :, :], axis=2)
    nearest_cluster_pos = np.argmin(dist, axis=1)
    nearest_cluster_ids = np.asarray([valid_clusters[int(i)] for i in nearest_cluster_pos], dtype=int)
    labels[noise_idx] = nearest_cluster_ids
    return labels, n_noise


def plot_clusters(df_plot, out_fig: Path):
    plt.figure(figsize=(9.5, 7.2))

    labels = df_plot["cluster"].values
    x = df_plot["UMAP1"].values
    y = df_plot["UMAP2"].values

    unique_labels = sorted(np.unique(labels))
    non_noise_labels = [lab for lab in unique_labels if lab != -1]
    # Use high-contrast colors for readability.
    strong_colors = [
        "#d62728",  # red
        "#1f77b4",  # blue
        "#2ca02c",  # green
        "#9467bd",  # purple
        "#ff7f0e",  # orange
        "#17becf",  # cyan
        "#8c564b",  # brown
        "#e377c2",  # pink
    ]
    color_map = {lab: strong_colors[i % len(strong_colors)] for i, lab in enumerate(non_noise_labels)}

    for lab in unique_labels:
        mask = labels == lab
        if lab == -1:
            plt.scatter(
                x[mask],
                y[mask],
                s=28,
                c="#9e9e9e",
                alpha=0.65,
                marker="x",
                label="Noise",
            )
        else:
            n_pts = int(np.sum(mask))
            point_size = 46 if n_pts <= 20 else 36
            plt.scatter(
                x[mask],
                y[mask],
                s=point_size,
                color=color_map[lab],
                alpha=0.92,
                edgecolors="white",
                linewidths=0.35,
                label=f"Cluster {lab}",
            )

    # Label each cluster outside the point cloud and connect to centroid.
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    x_span = max(x_max - x_min, 1e-8)
    y_span = max(y_max - y_min, 1e-8)

    for lab in unique_labels:
        if lab == -1:
            continue
        m = df_plot["cluster"] == lab
        pts = df_plot.loc[m, ["UMAP1", "UMAP2"]].values.astype(np.float64)
        # Use medoid (real point nearest centroid) for stable annotation anchor.
        centroid = pts.mean(axis=0)
        d = np.linalg.norm(pts - centroid, axis=1)
        medoid_idx = int(np.argmin(d))
        cx, cy = float(pts[medoid_idx, 0]), float(pts[medoid_idx, 1])
        n = int(m.sum())
        # Move text outward based on cluster position to reduce overlap.
        sign_x = 1.0 if cx >= (x_min + x_max) / 2.0 else -1.0
        sign_y = 1.0 if cy >= (y_min + y_max) / 2.0 else -1.0
        tx = cx + sign_x * 0.10 * x_span
        ty = cy + sign_y * 0.12 * y_span

        # Clamp text position inside axes with a small margin.
        tx = float(np.clip(tx, x_min + 0.03 * x_span, x_max - 0.03 * x_span))
        ty = float(np.clip(ty, y_min + 0.04 * y_span, y_max - 0.04 * y_span))

        plt.annotate(
            f"C{lab} (n={n})",
            xy=(cx, cy),
            xytext=(tx, ty),
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="black", lw=0.7, alpha=0.92),
            arrowprops=dict(arrowstyle="-", color="black", lw=0.8, alpha=0.85),
        )

    plt.title("Protein Sequence Clustering of High kcat/Km Candidates", fontsize=14)
    plt.xlabel("UMAP-1", fontsize=12)
    plt.ylabel("UMAP-2", fontsize=12)
    plt.grid(alpha=0.20, linestyle="--")
    plt.legend(frameon=True, fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(out_fig, dpi=600, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    input_path = Path(args.input).resolve()
    out_csv = Path(args.output_csv).resolve()
    out_fig = Path(args.output_fig).resolve()
    out_metrics = Path(args.output_metrics).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    df = read_csv_auto(input_path).copy()
    if args.seq_col not in df.columns:
        raise ValueError(f"Sequence column `{args.seq_col}` not found. Available columns: {list(df.columns)}")

    sequences = df[args.seq_col].astype(str).str.strip().tolist()
    print(f"Total sequences: {len(sequences)}")
    print("Extracting ESMC embeddings ...")
    emb = get_protein_embeddings(sequences)
    emb = StandardScaler().fit_transform(emb)

    print("Clustering ...")
    labels_raw, method = run_hdbscan_or_fallback(
        emb,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        seed=args.seed,
    )
    if args.keep_noise:
        labels = labels_raw
        reassigned_noise = 0
    else:
        labels, reassigned_noise = assign_noise_to_nearest_cluster(emb, labels_raw)
        if reassigned_noise > 0:
            print(f"Noise reassigned to nearest clusters: {reassigned_noise}")
    df["cluster"] = labels

    print("Running UMAP for visualization ...")
    xy = run_umap(
        emb,
        labels=labels,
        seed=args.seed,
        auto_tune=args.auto_umap_tune,
        label_guided=args.label_guided_umap,
    )
    df["UMAP1"] = xy[:, 0]
    df["UMAP2"] = xy[:, 1]

    metrics = compute_clustering_metrics(embeddings=emb, umap_xy=xy, labels=labels)

    # Sort by cluster and activity ratio for easier inspection.
    if "Pred_kcat_over_Km" in df.columns:
        df["Pred_kcat_over_Km"] = pd.to_numeric(df["Pred_kcat_over_Km"], errors="coerce")
        df = df.sort_values(["cluster", "Pred_kcat_over_Km"], ascending=[True, False]).reset_index(drop=True)
    elif "kcat_over_Km" in df.columns:
        df["kcat_over_Km"] = pd.to_numeric(df["kcat_over_Km"], errors="coerce")
        df = df.sort_values(["cluster", "kcat_over_Km"], ascending=[True, False]).reset_index(drop=True)
    else:
        df = df.sort_values(["cluster"]).reset_index(drop=True)

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame([metrics]).to_csv(out_metrics, index=False, encoding="utf-8-sig")
    plot_clusters(df, out_fig)

    n_cluster = len([x for x in np.unique(labels) if x != -1])
    n_noise = int(np.sum(labels == -1))
    print(f"Method: {method}")
    print(f"Clusters (excluding noise): {n_cluster}")
    print(f"Noise points: {n_noise}")
    print(f"Clustered CSV saved: {out_csv}")
    print(f"Metrics CSV saved: {out_metrics}")
    print(f"Figure saved (600 dpi): {out_fig}")
    print("\nClustering metrics:")
    for k in [
        "n_total",
        "n_clusters_excluding_noise",
        "n_noise",
        "noise_ratio",
        "silhouette_umap",
        "davies_bouldin_umap",
        "calinski_harabasz_umap",
        "silhouette_embedding",
        "davies_bouldin_embedding",
        "calinski_harabasz_embedding",
        "separation_ratio_umap",
        "largest_to_smallest_ratio",
        "shannon_entropy_normalized",
        "effective_number_of_clusters",
    ]:
        print(f"{k}: {metrics.get(k)}")


if __name__ == "__main__":
    main()

