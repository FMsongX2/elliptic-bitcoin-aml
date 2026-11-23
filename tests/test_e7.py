"""Tests for E7 semi-supervised / label efficiency (setting B)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common
import e7_semi_supervised as e7
from test_e6b import make_synthetic_dataset


def _toy(n_pool=400, n_unk=300, d=57, seed=0):
    rng = np.random.RandomState(seed)
    y = (rng.rand(n_pool) < 0.15).astype(int)
    Xp = rng.randn(n_pool, d) + y[:, None] * 1.5
    Xu = rng.randn(n_unk, d)
    X = np.vstack([Xp, Xu]).astype(np.float32)
    return X, np.arange(n_pool), y, np.arange(n_pool, n_pool + n_unk)


@pytest.mark.parametrize("method", e7.METHODS)
def test_methods_respect_label_budget(method):
    X, pool, y, unk = _toy()
    model, n_lab, n_ill = e7.train_method(method, X, pool, y, unk, budget=40, seed=0, emb_cache={})
    assert n_lab == 40, f"{method} used {n_lab} labels, budget 40"
    assert 0 <= n_ill <= n_lab
    p = model.predict_proba(X[:10])[:, 1]
    assert p.shape == (10,) and np.all((p >= 0) & (p <= 1))


def test_single_class_labels_fall_back_to_prior():
    X, pool, y, unk = _toy()
    y0 = np.zeros_like(y)
    model, _, n_ill = e7.train_method("sup_random", X, pool, y0, unk, budget=30, seed=0, emb_cache={})
    assert n_ill == 0
    assert np.allclose(model.predict_proba(X[:5])[:, 1], 0.0)


def test_e7_fast_synthetic(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    results_dir, fig_dir, model_dir = tmp_path / "results", tmp_path / "fig", tmp_path / "model"
    make_synthetic_dataset(data_dir)
    monkeypatch.setattr(common, "DATA_DIR", data_dir)
    monkeypatch.setattr(common, "RESULTS_PATH", results_dir / "results.csv")
    monkeypatch.setattr(e7, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(e7, "FIG_DIR", fig_dir)
    monkeypatch.setattr(e7, "MODEL_DIR", model_dir)
    monkeypatch.setattr(e7, "MIN_LABELS", 10)

    used_cols = []
    orig = e7._fit_split

    def spy(df, steps, cols):
        used_cols.append(list(cols))
        assert max(steps) <= 34, "fit steps must not include test steps"
        return orig(df, steps, cols)

    monkeypatch.setattr(e7, "_fit_split", spy)
    assert e7.main(["--fast"]) == 0

    allowed = set(common.FEATURE_SETS["time"])
    for cols in used_cols:
        assert set(cols) <= allowed, "only setting-B (transaction-time) features allowed"
        assert not any(c in cols for c in [f"local_{i}" for i in range(21, 93)])

    runs = pd.read_csv(results_dir / "e7_runs.csv")
    assert set(runs["method"]) == set(e7.METHODS)
    assert set(runs.loc[runs["method"].isin(e7.SMALL_ONLY), "frac"]) == {0.10}
    for f in ["e7_label_curve_pr_auc.png", "e7_label_curve_f1.png", "e7_period.png"]:
        assert (fig_dir / f).exists(), f
    res = pd.read_csv(results_dir / "results.csv", encoding="utf-8-sig")
    e7_rows = res[res["experiment"] == "E7"]
    assert len(e7_rows) == len(runs)
    assert set(e7_rows["feature_set"]) <= {"time", "time-tp"}
    assert e7_rows["note"].str.contains("frac=").all()


def test_label_spreading_unreached_rows_get_prior_not_nan():
    """Points whose neighbours got no label must not yield NaN probabilities."""
    rng = np.random.RandomState(0)
    # two far-apart blobs; labels only in blob A -> blob B is unreached in the kNN graph
    A = rng.randn(300, 57)
    B = rng.randn(300, 57) + 50
    X = np.vstack([A, B]).astype(np.float32)
    y = (rng.rand(300) < 0.3).astype(int)
    model, _, _ = e7.train_method("label_spreading", X, np.arange(300), y, np.arange(300, 600),
                                  budget=60, seed=0, emb_cache={})
    p = model.predict_proba(B[:50].astype(np.float32))
    assert np.isfinite(p).all()
