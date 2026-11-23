"""Fast unit tests for Preprocessing Diagnostics on synthetic data."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

# Ensure the repo root is on sys.path
SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import common
import prep_diagnostics


def make_minimal_synthetic_df() -> pd.DataFrame:
    """Generate a fast, deterministic synthetic DataFrame with 165 features."""
    rng = np.random.RandomState(42)
    n_rows = 150
    # Steps across 1..49
    steps = [(i % 49) + 1 for i in range(n_rows)]
    tx_ids = list(range(1001, 1001 + n_rows))

    # Base random Gaussian features
    data = rng.randn(n_rows, 165).astype(np.float32)

    # Make local_0 near-constant: 96% same value
    data[: int(0.96 * n_rows), 0] = 1.0

    # Make local_1 and local_2 perfectly correlated
    data[:, 2] = data[:, 1] + 1e-6

    # Make local_3 have an extreme heavy-tail outlier
    data[0, 3] = 50.0

    # Exact duplicate row at row 10 and 11
    data[11] = data[10]

    cols = ["txId", "time_step"] + [f"local_{i}" for i in range(93)] + [f"agg_{i}" for i in range(72)]
    df = pd.DataFrame(data, columns=[f"local_{i}" for i in range(93)] + [f"agg_{i}" for i in range(72)])
    df.insert(0, "time_step", steps)
    df.insert(0, "txId", tx_ids)

    # Labels: illicit (1), licit (0), or unknown (NaN)
    labels = []
    for i, s in enumerate(steps):
        if s <= 30 and i % 3 == 0:
            labels.append(np.nan)
        elif i % 2 == 0:
            labels.append(1.0)
        else:
            labels.append(0.0)
    df["y"] = labels

    return df


def test_compute_feature_stats():
    """Verify skewness, kurtosis, mode share, and extreme z-score calculation."""
    df = make_minimal_synthetic_df()
    train_mask = df["time_step"].isin(common.TRAIN_STEPS)
    df_train = df[train_mask]
    features = common.feature_cols(df, "all")

    stats_df = prep_diagnostics.compute_feature_stats(df_train, features)
    assert len(stats_df) == 165
    assert set(stats_df.columns) == {
        "feature", "feature_type", "skewness", "excess_kurtosis",
        "most_frequent_value_share", "n_unique", "fraction_abs_z_gt_5",
    }

    # local_0 should have very high mode share (> 0.90)
    row_local0 = stats_df[stats_df["feature"] == "local_0"].iloc[0]
    assert row_local0["most_frequent_value_share"] > 0.90

    # Near-constant detection
    near_const = prep_diagnostics.find_near_constant_features(stats_df, threshold=0.90)
    assert "local_0" in near_const


def test_compute_duplicates():
    """Verify duplicate row count and near-duplicate share."""
    df = make_minimal_synthetic_df()
    train_mask = df["time_step"].isin(common.TRAIN_STEPS)
    df_train = df[train_mask]
    features = common.feature_cols(df, "all")

    exact_count, near_share = prep_diagnostics.compute_duplicates(df_train, features, decimals=4)
    assert exact_count >= 1
    assert 0.0 <= near_share <= 1.0


def test_correlation_structure():
    """Verify high correlation pair counting and greedy collinear grouping."""
    df = make_minimal_synthetic_df()
    train_mask = df["time_step"].isin(common.TRAIN_STEPS)
    df_train = df[train_mask]
    features = common.feature_cols(df, "all")

    corr, pair_counts, greedy_groups = prep_diagnostics.compute_correlation_structure(df_train, features)
    assert corr.shape == (165, 165)
    assert pair_counts["pairs_gt_0.95"] >= 1
    assert pair_counts["pairs_gt_0.99"] >= 1

    # local_1 and local_2 should be in a greedy group together
    found_pair = any("local_1" in g and "local_2" in g for g in greedy_groups)
    assert found_pair


def test_compute_drift_and_posthoc():
    """Verify KS drift across train/test and posthoc label KS."""
    df = make_minimal_synthetic_df()
    features = common.feature_cols(df, "all")

    drift_df = prep_diagnostics.compute_drift(df, features)
    assert len(drift_df) == 165
    assert "ks_train_test" in drift_df.columns
    assert "ks_pre_post_dark" in drift_df.columns
    assert (drift_df["ks_train_test"] >= 0.0).all()
    assert (drift_df["ks_train_test"] <= 1.0).all()

    train_mask = df["time_step"].isin(common.TRAIN_STEPS)
    posthoc_df = prep_diagnostics.compute_posthoc_ks(df[train_mask], features)
    assert len(posthoc_df) == 165
    assert "ks_posthoc_illicit_licit" in posthoc_df.columns
    assert (posthoc_df["ks_posthoc_illicit_licit"] >= 0.0).all()


def test_run_diagnostics_end_to_end_synthetic(tmp_path: Path):
    """Verify full diagnostic workflow creates all required files and figures."""
    df = make_minimal_synthetic_df()
    out_dir = tmp_path / "results"
    fig_dir = tmp_path / "figures"

    summary = prep_diagnostics.run_diagnostics(df=df, output_dir=out_dir, figure_dir=fig_dir)

    assert (out_dir / "prep_feature_stats.csv").exists()
    assert (out_dir / "prep_drift.csv").exists()
    assert (out_dir / "prep_summary.json").exists()
    assert (fig_dir / "prep_skew_kurtosis.png").exists()
    assert (fig_dir / "prep_corr_heatmap.png").exists()
    assert (fig_dir / "prep_drift_top.png").exists()

    assert summary["n_features_total"] == 165
    assert summary["dataset_rows_total"] == len(df)
    assert summary["runtime_seconds"] > 0
