"""Tests for E7c (rolling-origin search)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common
import e7c_search as e7c
from test_e6b import make_synthetic_dataset


def test_folds_are_forward_in_time():
    for name, fit, ev in e7c.FOLDS + [e7c.TEST]:
        assert max(fit) < min(ev), name
    assert max(e7c.FOLDS[-1][2]) <= 34 and min(e7c.TEST[2]) == 35


def test_recency_weight_halves_every_half_life():
    w = e7c.recency(np.array([30, 25, 20]), 30, 5)
    assert np.allclose(w, [1.0, 0.5, 0.25])
    assert np.allclose(e7c.recency(np.array([1, 30]), 30, None), 1.0)


@pytest.fixture
def data(tmp_path, monkeypatch):
    make_synthetic_dataset(tmp_path / "data")
    monkeypatch.setattr(common, "DATA_DIR", tmp_path / "data")
    df = common.load_dataset(labeled_only=False)
    big = pd.concat([df.assign(txId=df.txId + 10**6 * k) for k in range(6)], ignore_index=True)
    rng = np.random.RandomState(0)
    big[common.feature_cols(big, "time")] += rng.randn(len(big), 57) * 0.1
    return e7c.Data(big)


@pytest.mark.parametrize("cfg", [
    dict(kind="sup", base="team", half_life=None, unsup=False),
    dict(kind="sup", base="balanced", half_life=5, unsup=True),
    dict(kind="st", base="small", half_life=10, unsup=False, thr=0.9, balanced=True, plw=0.5),
    dict(kind="ls"),
])
def test_run_returns_probabilities_for_eval_rows_only(cfg, data):
    p, ev, n_lab, n_ill = e7c.run(cfg, data, e7c.FOLDS[2], 0.1, 0)
    assert len(p) == len(ev) and np.all((p >= 0) & (p <= 1))
    assert set(data.t[ev]) <= set(range(31, 35))
    assert n_lab >= e7c.MIN_LABELS


def test_unsup_scores_ignore_eval_rows(data):
    fit, _, _, ev = data.split(range(1, 31), range(31, 35))
    f = data.unsup_scores("k", fit)
    data.X[ev] += 1000.0                  # change eval rows, then fit again: must not matter
    f2 = data.unsup_scores("k2", fit)
    data.X[ev] -= 1000.0
    assert np.allclose(f(ev), f2(ev), atol=1e-4)


def test_main_stage1_small(tmp_path, monkeypatch):
    make_synthetic_dataset(tmp_path / "data")
    monkeypatch.setattr(common, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(e7c, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(e7c, "FRACS", [0.1])
    monkeypatch.setattr(e7c, "SEL_SEEDS", [0])
    monkeypatch.setattr(e7c, "TEST_SEEDS", [0])
    monkeypatch.setattr(e7c, "MIN_LABELS", 10)
    monkeypatch.setattr(e7c, "stage1_configs", lambda: [dict(kind="sup", base="team", half_life=None, unsup=False),
                                                        dict(kind="sup", base="small", half_life=5, unsup=True)])
    assert e7c.main(["--stage", "1", "--fast"]) == 0
    runs = pd.read_csv(tmp_path / "e7c_runs.csv")
    assert len(runs) == 2 * 4 and set(runs.split) == {"cv1", "cv2", "cv3", "test"}
    assert (tmp_path / "e7c_preds.joblib").exists()
