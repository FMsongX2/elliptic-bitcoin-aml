"""Tests for E7b (embedding choice + realistic threshold)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common
import e7_semi_supervised as e7
import e7b_semi_supervised as e7b
from test_e6b import make_synthetic_dataset


def test_realistic_threshold_uses_label_share_and_fallback():
    rng = np.random.RandomState(0)
    y = (rng.rand(3000) < 0.17).astype(int)
    p = np.clip(y * 0.6 + rng.rand(3000) * 0.5, 0, 1)
    thr, n = e7b.realistic_threshold(y, p, 0.01, seed=0)
    assert n == 30 and 0.05 <= thr <= 0.95
    _, n_min = e7b.realistic_threshold(y, p, 0.001, seed=0)
    assert n_min == e7b.MIN_VAL_LABELS
    _, n_all = e7b.realistic_threshold(y, p, 1.0, seed=0)
    assert n_all == 3000
    thr0, _ = e7b.realistic_threshold(np.zeros(50, int), np.linspace(0, 1, 50), 0.5, seed=0)
    assert thr0 == 0.5


def test_configs_cross_distance_methods_with_embeddings():
    c = e7b.configs(["sup_random", "label_spreading"], ["quantile_57", "quantile_54"])
    assert c == [("sup_random", "-"), ("label_spreading", "quantile_57"), ("label_spreading", "quantile_54")]


def test_e7b_fast_synthetic(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    make_synthetic_dataset(data_dir)
    monkeypatch.setattr(common, "DATA_DIR", data_dir)
    monkeypatch.setattr(common, "RESULTS_PATH", tmp_path / "results.csv")
    monkeypatch.setattr(e7b, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(e7b, "FIG_DIR", tmp_path)
    monkeypatch.setattr(e7, "MIN_LABELS", 10)
    assert e7b.main(["--fast", "--methods", "sup_random", "label_spreading", "cluster_rep",
                     "--fracs", "0.1", "--seeds", "0"]) == 0
    runs = pd.read_csv(tmp_path / "e7b_runs.csv")
    assert set(runs.embedding[runs.method == "label_spreading"]) == set(e7b.EMBEDDINGS)
    assert (runs.embedding[runs.method == "sup_random"] == "-").all()
    assert runs[["f1_full", "f1_real", "pr_auc"]].notna().all().all()
    res = pd.read_csv(tmp_path / "results.csv", encoding="utf-8-sig")
    assert (res.experiment == "E7b").sum() == len(runs)
    assert (tmp_path / "e7b_embedding.png").exists() and (tmp_path / "e7b_threshold.png").exists()
