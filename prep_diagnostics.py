"""Data Preprocessing Diagnostics for Elliptic Bitcoin Dataset (Epoch 2).

This script performs label-free diagnostics on the 165 features of the Elliptic
dataset, followed by a clearly separated post-hoc label analysis:
1. Feature statistics: skewness, excess kurtosis, mode share, unique count, |z|>5 fraction.
2. Near-constant features, exact duplicates, and near-duplicates (round 4).
3. Correlation structure: pairs with |r|>0.95, >0.99, and greedy collinear groups.
4. Temporal distribution shift (drift): Kolmogorov-Smirnov test between Train (1-30)
   and Test (35-49), and across the dark market shutdown (35-42 vs 43-49).
5. Post-hoc analysis (labeled train): KS statistic between illicit and licit transactions.
6. Diagnostic figures (English, Agg, dpi 130) and JSON summary.

Usage:
    python prep_diagnostics.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import cluster, stats
from sklearn.preprocessing import StandardScaler

# Ensure the repo root is on sys.path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common import BASE
import common


# ==============================================================================
# 1. Feature Distribution Statistics (Label-Free, Train Rows 1-30)
# ==============================================================================

def compute_feature_stats(df_train: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Compute per-feature distribution statistics on train rows (steps 1-30).

    Returns a DataFrame with columns:
        - feature: feature name
        - feature_type: 'local' or 'agg'
        - skewness: Fisher-Pearson skewness
        - excess_kurtosis: Fisher excess kurtosis (normal distribution = 0)
        - most_frequent_value_share: frequency of the mode value / total train rows
        - n_unique: number of distinct values
        - fraction_abs_z_gt_5: fraction of samples with |z| > 5 after StandardScaler
    """
    X = df_train[features].values.astype(np.float64)
    n_samples, n_feats = X.shape

    # Skewness and excess kurtosis (Fisher, unbiased)
    skews = stats.skew(X, axis=0, bias=False)
    kurts = stats.kurtosis(X, axis=0, fisher=True, bias=False)

    # Standardize on train to measure extreme tails (|z| > 5)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    frac_z5 = (np.abs(X_scaled) > 5.0).mean(axis=0)

    rows = []
    for j, feat in enumerate(features):
        col = X[:, j]
        # Fast mode share and nunique
        unq, counts = np.unique(col, return_counts=True)
        mode_share = float(counts.max()) / float(n_samples)
        n_unq = int(len(unq))

        rows.append({
            "feature": feat,
            "feature_type": "local" if feat.startswith("local_") else "agg",
            "skewness": float(skews[j]),
            "excess_kurtosis": float(kurts[j]),
            "most_frequent_value_share": round(mode_share, 6),
            "n_unique": n_unq,
            "fraction_abs_z_gt_5": round(float(frac_z5[j]), 6),
        })

    return pd.DataFrame(rows)


def find_near_constant_features(stats_df: pd.DataFrame, threshold: float = 0.95) -> List[str]:
    """Identify features where the most frequent value accounts for > threshold of rows."""
    mask = stats_df["most_frequent_value_share"] > threshold
    return stats_df.loc[mask, "feature"].tolist()


# ==============================================================================
# 2. Row Duplicates & Near-Duplicates (Label-Free, Train Rows 1-30)
# ==============================================================================

def compute_duplicates(df_train: pd.DataFrame, features: List[str], decimals: int = 4) -> Tuple[int, float]:
    """Calculate exact duplicate row count and near-duplicate share (rounded to decimals)."""
    sub_df = df_train[features]
    exact_count = int(sub_df.duplicated().sum())
    near_share = float(sub_df.round(decimals).duplicated().mean())
    return exact_count, round(near_share, 4)


# ==============================================================================
# 3. Correlation Structure & Greedy Collinear Grouping (Train Rows 1-30)
# ==============================================================================

def compute_correlation_structure(
    df_train: pd.DataFrame,
    features: List[str],
    thresholds: Tuple[float, ...] = (0.95, 0.99),
) -> Tuple[np.ndarray, Dict[str, int], List[List[str]]]:
    """Compute Pearson correlation matrix, pair counts over thresholds, and greedy collinear groups.

    Greedy grouping algorithm (|r| > 0.95):
        1. Form an adjacency graph where an edge exists if |r_{i, j}| > 0.95 (i != j).
        2. In each iteration, select the unassigned feature with the maximum degree within
           unassigned nodes (tie-break by lowest index).
        3. Form a group with this leader and all unassigned neighbors having |r| > 0.95.
        4. Remove all group members from unassigned.
        5. Stop when no edges exist between remaining unassigned nodes.
    """
    X = df_train[features].values.astype(np.float64)
    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)

    n_feats = len(features)
    triu_idx = np.triu_indices(n_feats, k=1)
    abs_r = np.abs(corr[triu_idx])

    pair_counts = {}
    for th in thresholds:
        pair_counts[f"pairs_gt_{th}"] = int((abs_r > th).sum())

    # Greedy grouping at |r| > 0.95
    adj = np.abs(corr) > 0.95
    np.fill_diagonal(adj, False)

    unassigned = set(range(n_feats))
    greedy_groups: List[List[str]] = []

    while True:
        degrees = {i: sum(adj[i, j] for j in unassigned if j != i) for i in unassigned}
        max_deg = max(degrees.values()) if degrees else 0
        if max_deg == 0:
            break

        # Pick leader with max degree among remaining unassigned
        leader = min(i for i, d in degrees.items() if d == max_deg)
        members = [j for j in unassigned if adj[leader, j]]
        group = [leader] + sorted(members)
        greedy_groups.append([features[i] for i in group])

        for m in group:
            unassigned.remove(m)

    return corr, pair_counts, greedy_groups


# ==============================================================================
# 4. Temporal Drift Diagnostics (Label-Free, All Rows)
# ==============================================================================

def compute_drift(df_all: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Compute 2-sample Kolmogorov-Smirnov statistics across time windows (all rows, unlabeled).

    Windows compared:
        - Train (steps 1-30) vs Test (steps 35-49)
        - Pre-Dark Market Test (steps 35-42) vs Post-Dark Market Test (steps 43-49)
    """
    train_vals = df_all[df_all["time_step"].isin(common.TRAIN_STEPS)][features].values
    test_vals = df_all[df_all["time_step"].isin(common.TEST_STEPS)][features].values

    pre_mask = df_all["time_step"].between(35, 42)
    post_mask = df_all["time_step"].between(43, 49)
    pre_vals = df_all[pre_mask][features].values
    post_vals = df_all[post_mask][features].values

    rows = []
    for j, feat in enumerate(features):
        s_tt, p_tt = stats.ks_2samp(train_vals[:, j], test_vals[:, j])
        s_pp, p_pp = stats.ks_2samp(pre_vals[:, j], post_vals[:, j])
        rows.append({
            "feature": feat,
            "feature_type": "local" if feat.startswith("local_") else "agg",
            "ks_train_test": round(float(s_tt), 6),
            "pvalue_train_test": float(p_tt),
            "ks_pre_post_dark": round(float(s_pp), 6),
            "pvalue_pre_post_dark": float(p_pp),
        })

    return pd.DataFrame(rows)


# ==============================================================================
# 5. Post-Hoc Label Analysis (Labeled Train Rows 1-30 Only)
# ==============================================================================

def compute_posthoc_ks(df_train: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Compute KS statistic between illicit (y=1) and licit (y=0) in the training set.

    NOTE: This is strictly post-hoc and only evaluated on labeled train rows (1-30).
    It is separated from label-free diagnostics to verify which features have inherent
    class-discriminative power.
    """
    labeled_train = df_train[df_train["y"].isin([0, 1])]
    illicit_vals = labeled_train[labeled_train["y"] == 1][features].values
    licit_vals = labeled_train[labeled_train["y"] == 0][features].values

    rows = []
    for j, feat in enumerate(features):
        s_label, p_label = stats.ks_2samp(illicit_vals[:, j], licit_vals[:, j])
        rows.append({
            "feature": feat,
            "ks_posthoc_illicit_licit": round(float(s_label), 6),
            "pvalue_posthoc_illicit_licit": float(p_label),
        })

    return pd.DataFrame(rows)


# ==============================================================================
# 6. Diagnostic Visualizations (English, Matplotlib Agg, dpi=130)
# ==============================================================================

def plot_skew_kurtosis(stats_df: pd.DataFrame, out_path: Path) -> None:
    """Plot skewness and excess kurtosis distributions and extreme outlier fractions."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=130)

    # Panel 1: Skewness vs Excess Kurtosis (Symmetric Log Scale for Kurtosis)
    ax1 = axes[0]
    local_mask = stats_df["feature_type"] == "local"
    agg_mask = stats_df["feature_type"] == "agg"

    ax1.scatter(
        stats_df.loc[local_mask, "skewness"],
        stats_df.loc[local_mask, "excess_kurtosis"],
        color="#1f77b4", alpha=0.7, edgecolors="none", s=36, label=f"Local ({int(local_mask.sum())})",
    )
    ax1.scatter(
        stats_df.loc[agg_mask, "skewness"],
        stats_df.loc[agg_mask, "excess_kurtosis"],
        color="#ff7f0e", alpha=0.7, edgecolors="none", s=36, label=f"Aggregate ({int(agg_mask.sum())})",
    )
    ax1.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax1.set_yscale("symlog", linthresh=10.0)
    ax1.set_xlabel("Fisher-Pearson Skewness", fontsize=11)
    ax1.set_ylabel("Excess Kurtosis (symlog scale)", fontsize=11)
    ax1.set_title("Feature Skewness vs Excess Kurtosis (Train 1-30)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="upper left")

    # Annotate top 5 kurtosis features
    top_kurt = stats_df.sort_values(by="excess_kurtosis", ascending=False).head(5)
    for _, row in top_kurt.iterrows():
        ax1.annotate(
            row["feature"],
            xy=(row["skewness"], row["excess_kurtosis"]),
            xytext=(row["skewness"] + 5, row["excess_kurtosis"] * 1.1),
            fontsize=8,
            fontweight="semibold",
            arrowprops=dict(arrowstyle="->", color="#333333", lw=0.6),
        )

    # Panel 2: Distribution of Fraction Beyond |z| > 5
    ax2 = axes[1]
    top_z5 = stats_df.sort_values(by="fraction_abs_z_gt_5", ascending=False).head(15)
    y_pos = np.arange(len(top_z5))
    colors = ["#1f77b4" if t == "local" else "#ff7f0e" for t in top_z5["feature_type"]]

    ax2.barh(y_pos, top_z5["fraction_abs_z_gt_5"] * 100, color=colors, alpha=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_z5["feature"], fontsize=9)
    ax2.invert_yaxis()  # highest at top
    ax2.set_xlabel("Percentage of Samples with |z| > 5 (%)", fontsize=11)
    ax2.set_title("Top 15 Extreme Outlier Features (|z| > 5)", fontsize=12, fontweight="bold")
    ax2.grid(True, axis="x", linestyle=":", alpha=0.5)

    # Add Gaussian reference note
    ax2.axvline(5.7e-5, color="red", linestyle=":", label="Normal Dist. Ref (approx. 0.00006%)")
    ax2.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_corr_heatmap(corr: np.ndarray, features: List[str], out_path: Path) -> None:
    """Plot feature correlation matrix ordered by hierarchical clustering."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=130)

    # Hierarchical clustering order based on correlation distance
    dist = 1.0 - np.abs(corr)
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, 2.0)
    # Convert square distance matrix to condensed form for linkage
    triu_i, triu_j = np.triu_indices(len(features), k=1)
    condensed_dist = dist[triu_i, triu_j]

    link = cluster.hierarchy.linkage(condensed_dist, method="average")
    order = cluster.hierarchy.leaves_list(link)

    ordered_corr = corr[order, :][:, order]

    im = ax.imshow(ordered_corr, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson Correlation r", fontsize=10)

    r_abs = np.abs(corr[np.triu_indices(len(features), k=1)])
    ax.set_title(
        f"Clustered Feature Correlation Heatmap ({len(features)} Features)\n"
        f"{int((r_abs > 0.95).sum())} pairs |r| > 0.95, {int((r_abs > 0.99).sum())} pairs |r| > 0.99",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Features (Hierarchical Clustering Order)", fontsize=10)
    ax.set_ylabel("Features (Hierarchical Clustering Order)", fontsize=10)

    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_drift_top(
    drift_df: pd.DataFrame,
    posthoc_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Plot top 15 drifting features comparing Train vs Test drift with post-hoc label KS."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged = drift_df.merge(posthoc_df, on="feature")
    top15 = merged.sort_values(by="ks_train_test", ascending=False).head(15).copy()

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=130)

    y = np.arange(len(top15))
    bar_height = 0.28

    r1 = ax.barh(y - bar_height, top15["ks_train_test"], height=bar_height,
                 color="#d62728", alpha=0.85, label="Drift: Train(1-30) vs Test(35-49)")
    r2 = ax.barh(y, top15["ks_pre_post_dark"], height=bar_height,
                 color="#ff7f0e", alpha=0.85, label="Drift: Pre-Dark(35-42) vs Post-Dark(43-49)")
    r3 = ax.barh(y + bar_height, top15["ks_posthoc_illicit_licit"], height=bar_height,
                 color="#2ca02c", alpha=0.85, label="Post-Hoc: Illicit vs Licit KS (Train)")

    ax.set_yticks(y)
    ax.set_yticklabels(top15["feature"], fontsize=9, fontweight="medium")
    ax.invert_yaxis()
    ax.set_xlabel("Kolmogorov-Smirnov Statistic (0 to 1)", fontsize=11)
    ax.set_xlim(0.0, 1.05)
    ax.set_title(
        "Top 15 Drifting Features (Aggregated Features Suffer Massive Drift)",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ==============================================================================
# 7. Orchestration & Main Pipeline
# ==============================================================================

def run_diagnostics(
    df: pd.DataFrame | None = None,
    output_dir: Path | None = None,
    figure_dir: Path | None = None,
    setting: str = "A",
) -> Dict[str, Any]:
    """Run full preprocessing diagnostics pipeline and save results & figures."""
    t0 = time.time()
    if df is None:
        df = common.load_dataset(labeled_only=False)

    if output_dir is None:
        output_dir = BASE / "results"
    if figure_dir is None:
        figure_dir = BASE / "figures"

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    features = common.feature_cols(df, "all" if setting == "A" else "time")
    pfx = "prep" if setting == "A" else "prep_B"  # setting B: 57 transaction-time features
    print(f"Setting {setting}: {len(features)} features")
    train_mask = df["time_step"].isin(common.TRAIN_STEPS)
    df_train = df[train_mask].copy()

    # 1. Feature Stats
    t_step = time.time()
    stats_df = compute_feature_stats(df_train, features)
    near_const_feats = find_near_constant_features(stats_df, threshold=0.95)
    stats_path = output_dir / f"{pfx}_feature_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"[{time.time()-t_step:.2f}s] Feature stats computed -> {stats_path}")

    # 2. Duplicate Rows
    t_step = time.time()
    exact_dupes, near_dupes_share = compute_duplicates(df_train, features, decimals=4)
    print(f"[{time.time()-t_step:.2f}s] Duplicates: exact={exact_dupes}, near_share={near_dupes_share:.4f}")

    # 3. Correlation Structure
    t_step = time.time()
    corr, pair_counts, greedy_groups = compute_correlation_structure(df_train, features)
    print(f"[{time.time()-t_step:.2f}s] Correlation: {pair_counts}, greedy groups={len(greedy_groups)}")

    # 4. Temporal Drift
    t_step = time.time()
    drift_df = compute_drift(df, features)
    print(f"[{time.time()-t_step:.2f}s] Drift computed")

    # 5. Post-Hoc Label Analysis
    t_step = time.time()
    posthoc_df = compute_posthoc_ks(df_train, features)
    print(f"[{time.time()-t_step:.2f}s] Post-hoc KS computed")

    # Merge drift and posthoc for unified drift CSV
    unified_drift = drift_df.merge(posthoc_df, on="feature")
    drift_path = output_dir / f"{pfx}_drift.csv"
    unified_drift.to_csv(drift_path, index=False)
    print(f"Unified drift stats saved -> {drift_path}")

    # 6. Figures
    t_step = time.time()
    fig_skew = figure_dir / f"{pfx}_skew_kurtosis.png"
    fig_corr = figure_dir / f"{pfx}_corr_heatmap.png"
    fig_drift = figure_dir / f"{pfx}_drift_top.png"

    plot_skew_kurtosis(stats_df, fig_skew)
    plot_corr_heatmap(corr, features, fig_corr)
    plot_drift_top(drift_df, posthoc_df, fig_drift)
    print(f"[{time.time()-t_step:.2f}s] Figures generated in {figure_dir}")

    # 7. Summary JSON
    top15_tt = drift_df.sort_values(by="ks_train_test", ascending=False).head(15)
    top15_pp = drift_df.sort_values(by="ks_pre_post_dark", ascending=False).head(15)
    top15_ph = posthoc_df.sort_values(by="ks_posthoc_illicit_licit", ascending=False).head(15)

    summary: Dict[str, Any] = {
        "dataset_rows_total": int(len(df)),
        "train_rows_total": int(len(df_train)),
        "val_rows_total": int(df["time_step"].isin(common.VAL_STEPS).sum()),
        "test_rows_total": int(df["time_step"].isin(common.TEST_STEPS).sum()),
        "test_pre_dark_rows": int(df["time_step"].between(35, 42).sum()),
        "test_post_dark_rows": int(df["time_step"].between(43, 49).sum()),
        "labeled_train_rows": int(df_train["y"].isin([0, 1]).sum()),
        "labeled_train_illicit": int((df_train["y"] == 1).sum()),
        "labeled_train_licit": int((df_train["y"] == 0).sum()),
        "n_features_total": len(features),
        "n_features_local": sum(1 for f in features if f.startswith("local_")),
        "n_features_agg": sum(1 for f in features if f.startswith("agg_")),
        "near_constant_features_count": len(near_const_feats),
        "near_constant_features": near_const_feats,
        "exact_duplicate_rows_count": exact_dupes,
        "near_duplicate_share_round_4": near_dupes_share,
        "correlation_pairs_gt_095": pair_counts["pairs_gt_0.95"],
        "correlation_pairs_gt_099": pair_counts["pairs_gt_0.99"],
        "greedy_collinear_groups_count": len(greedy_groups),
        "greedy_collinear_total_features": sum(len(g) for g in greedy_groups),
        "skewness_min": round(float(stats_df["skewness"].min()), 4),
        "skewness_median": round(float(stats_df["skewness"].median()), 4),
        "skewness_max": round(float(stats_df["skewness"].max()), 4),
        "excess_kurtosis_min": round(float(stats_df["excess_kurtosis"].min()), 4),
        "excess_kurtosis_median": round(float(stats_df["excess_kurtosis"].median()), 4),
        "excess_kurtosis_max": round(float(stats_df["excess_kurtosis"].max()), 4),
        "features_with_z5_outliers": int((stats_df["fraction_abs_z_gt_5"] > 0).sum()),
        "max_fraction_z5": round(float(stats_df["fraction_abs_z_gt_5"].max()), 6),
        "top_15_drifting_train_test": [
            {"feature": row["feature"], "ks": row["ks_train_test"]}
            for _, row in top15_tt.iterrows()
        ],
        "top_15_drifting_pre_post_dark": [
            {"feature": row["feature"], "ks": row["ks_pre_post_dark"]}
            for _, row in top15_pp.iterrows()
        ],
        "top_15_posthoc_illicit_licit_ks": [
            {"feature": row["feature"], "ks": row["ks_posthoc_illicit_licit"]}
            for _, row in top15_ph.iterrows()
        ],
        "runtime_seconds": round(time.time() - t0, 2),
    }

    summary_path = output_dir / f"{pfx}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary JSON saved -> {summary_path}")
    print(f"All diagnostics completed in {summary['runtime_seconds']}s")

    return summary


def main():
    print("=" * 70)
    print("Running Preprocessing Diagnostics for Epoch 2 (Elliptic Dataset)")
    print("=" * 70)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", choices=["A", "B"], default="A")
    run_diagnostics(setting=ap.parse_args().setting)


if __name__ == "__main__":
    main()
