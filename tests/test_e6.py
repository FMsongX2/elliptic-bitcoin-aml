"""Tests for E6 Unsupervised learning experiment."""
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
import detectors
import e6_unsupervised


def make_synthetic_dataset(data_dir: Path):
    """Generate minimal synthetic elliptic CSVs with 165 features."""
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(42)

    rows = []
    classes = []
    tx_id = 1000

    # Steps 1..30 (train), 31..34 (val), 35..42 (test early), 43..49 (test late)
    for step in range(1, 50):
        # 3 rows per step for train, 5 for val, 4 for test
        n_step = 5 if 31 <= step <= 34 else 4 if step >= 35 else 3
        for i in range(n_step):
            tx_id += 1
            feats = rng.randn(165).astype(np.float32)
            row = [tx_id, step] + feats.tolist()
            rows.append(row)

            # Assign labels: alternate 1 and 2, with some unknowns in train
            if step <= 30 and i == 0:
                cls = "unknown"
            elif i % 2 == 0:
                cls = "1"  # illicit
            else:
                cls = "2"  # licit
            classes.append({"txId": tx_id, "class": cls})

    feat_df = pd.DataFrame(rows)
    feat_df.to_csv(data_dir / "elliptic_txs_features.csv", header=False, index=False)

    cls_df = pd.DataFrame(classes)
    cls_df.to_csv(data_dir / "elliptic_txs_classes.csv", header=True, index=False)


def test_to_prob_monotone_and_range():
    """Requirement (a): to_prob is monotone and in [0, 1]."""
    fit_scores = np.array([-2.5, -1.0, 0.0, 0.5, 1.2, 3.0, 5.0])
    test_scores = np.array([-10.0, -2.5, -1.5, 0.0, 0.2, 0.5, 2.0, 5.0, 10.0])

    probs = e6_unsupervised.to_prob(fit_scores, test_scores)

    # 1. Range in [0, 1]
    assert np.all(probs >= 0.0), f"Found prob < 0: {probs[probs < 0]}"
    assert np.all(probs <= 1.0), f"Found prob > 1: {probs[probs > 1]}"

    # 2. Monotonicity: non-decreasing
    assert np.all(np.diff(probs) >= 0.0), f"Probabilities not monotonic: {probs}"

    # 3. Extremes
    assert e6_unsupervised.to_prob(fit_scores, np.array([-100.0]))[0] == 0.0
    assert e6_unsupervised.to_prob(fit_scores, np.array([100.0]))[0] == 1.0


def test_e6_fast_synthetic(tmp_path, monkeypatch):
    """Requirement (b): --fast run on synthetic data writes all outputs."""
    data_dir = tmp_path / "data" / "elliptic"
    results_dir = tmp_path / "results"
    fig_dir = tmp_path / "report" / "figures"
    model_dir = tmp_path / "model"

    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    make_synthetic_dataset(data_dir)

    results_csv = results_dir / "results.csv"

    # Monkeypatch paths in common, paths, and e6_unsupervised
    monkeypatch.setattr(common, "DATA_DIR", data_dir)
    monkeypatch.setattr(common, "RESULTS_PATH", results_csv)
    monkeypatch.setattr(e6_unsupervised, "FIG_DIR", fig_dir)
    monkeypatch.setattr(e6_unsupervised, "MODEL_DIR", model_dir)
    monkeypatch.setattr(e6_unsupervised, "RESULTS_DIR", results_dir)

    # Run e6_unsupervised with --fast
    ret = e6_unsupervised.main(["--fast"])
    assert ret == 0

    # 1. Check results files
    expected_result_files = ["e6_grid.csv", "e6_summary.csv", "e6_by_step.csv", "e6_intrinsic.csv"]
    for fname in expected_result_files:
        p = results_dir / fname
        assert p.exists(), f"Missing result file: {p}"
        assert p.stat().st_size > 0, f"Empty result file: {p}"

    # 2. Check 4 figures
    expected_figures = [
        "e6_direction.png",
        "e6_pr_curves.png",
        "e6_by_step.png",
        "e6_kmeans_clusters.png",
    ]
    for fname in expected_figures:
        p = fig_dir / fname
        assert p.exists(), f"Missing figure: {p}"
        assert p.stat().st_size > 0, f"Empty figure: {p}"

    # 3. Check 10 models (5 groups x 2 modes) + ref files + manifest
    expected_models = [
        f"{mode}_{group}.joblib"
        for mode in ("best", "anomaly")
        for group in ("kmeans", "dbscan", "gmm", "iforest", "lof")
    ]
    for fname in expected_models:
        p = model_dir / fname
        assert p.exists(), f"Missing model: {p}"
        assert p.stat().st_size > 0, f"Empty model: {p}"

        # Ref npy
        ref_p = model_dir / fname.replace(".joblib", "_ref.npy")
        assert ref_p.exists(), f"Missing ref npy: {ref_p}"
        assert ref_p.stat().st_size > 0, f"Empty ref npy: {ref_p}"

    manifest_p = model_dir / "manifest.json"
    assert manifest_p.exists(), "Missing manifest.json"
    manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
    assert "versions" in manifest
    assert "seeds" in manifest
    assert manifest["seeds"] == [0, 1, 2]
    assert len(manifest["models"]) == 10

    # Verify dropped _fit_X and sorted ref arrays
    import joblib
    for fname in expected_models:
        det = joblib.load(model_dir / fname)
        assert getattr(det, "_fit_X", None) is None, f"_fit_X was not dropped for {fname}"
        ref_arr = np.load(model_dir / fname.replace(".joblib", "_ref.npy"))
        assert len(ref_arr) > 0, f"Ref array is empty for {fname}"
        assert np.all(np.diff(ref_arr) >= 0.0), f"Ref array is not sorted for {fname}"

    # Verify summary table
    summary_df = pd.read_csv(results_dir / "e6_summary.csv")
    assert len(summary_df) == 10, f"Expected 10 rows in e6_summary.csv, got {len(summary_df)}"
    for col in ["mode", "model", "params", "direction", "val_pr_auc", "pr_auc_mean", "pr_auc_std"]:
        assert col in summary_df.columns, f"Missing {col} in e6_summary.csv"

    # Verify by_step table
    by_step_df = pd.read_csv(results_dir / "e6_by_step.csv")
    assert len(by_step_df) > 0, "e6_by_step.csv is empty"
    assert "time_step" in by_step_df.columns
    assert (by_step_df["time_step"] >= 35).all(), "e6_by_step should only contain test steps (>=35)"

    # Verify intrinsic table
    intrinsic_df = pd.read_csv(results_dir / "e6_intrinsic.csv")
    assert len(intrinsic_df) >= 3, "Expected at least 3 rows in e6_intrinsic.csv (kmeans, dbscan, gmm)"
    assert set(intrinsic_df["group"].unique()).issubset({"kmeans", "dbscan", "gmm", "iforest", "lof"})

    # 4. Check results.csv for E6 entries
    assert results_csv.exists(), "Missing results/results.csv"
    res_df = pd.read_csv(results_csv)
    e6_rows = res_df[res_df["experiment"] == "E6"]
    assert len(e6_rows) == 30, f"Expected 30 E6 rows (10 models x 3 seeds), got {len(e6_rows)}"
    assert set(e6_rows["seed"].unique()) == {0, 1, 2}
    assert (e6_rows["feature_set"] == "all").all()
    for _, row in e6_rows.iterrows():
        note = str(row["note"])
        assert "params=" in note
        assert "direction=" in note
        assert "pr_auc_35_42=" in note
        assert "pr_auc_43_49=" in note
        assert "threshold=tuned_on_val" in note


def test_fit_rows_grid_stage_no_future(tmp_path, monkeypatch):
    """Requirement (c): fit rows never include time_step > 30 in grid stage."""
    data_dir = tmp_path / "data" / "elliptic"
    results_dir = tmp_path / "results"
    fig_dir = tmp_path / "report" / "figures"
    model_dir = tmp_path / "model"

    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    make_synthetic_dataset(data_dir)

    monkeypatch.setattr(common, "DATA_DIR", data_dir)
    monkeypatch.setattr(common, "RESULTS_PATH", results_dir / "results.csv")
    monkeypatch.setattr(e6_unsupervised, "FIG_DIR", fig_dir)
    monkeypatch.setattr(e6_unsupervised, "MODEL_DIR", model_dir)
    monkeypatch.setattr(e6_unsupervised, "RESULTS_DIR", results_dir)

    ret = e6_unsupervised.main(["--fast"])
    assert ret == 0

    # Check recorded attribute
    assert hasattr(e6_unsupervised, "GRID_FIT_MAX_STEP")
    assert e6_unsupervised.GRID_FIT_MAX_STEP <= 30

    # Check e6_grid.csv
    grid_df = pd.read_csv(results_dir / "e6_grid.csv")
    assert "fit_max_t" in grid_df.columns
    assert (grid_df["fit_max_t"] <= 30).all(), f"Found fit_max_t > 30: {grid_df['fit_max_t'].max()}"
