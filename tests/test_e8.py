"""Tests for E8 (recovery after the step-43 shift)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common
import e8_shift_recovery as e8
from test_e6b import make_synthetic_dataset


def test_e8_never_trains_on_test_steps(tmp_path, monkeypatch):
    make_synthetic_dataset(tmp_path / "data")
    monkeypatch.setattr(common, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(common, "RESULTS_PATH", tmp_path / "results.csv")
    monkeypatch.setattr(e8, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(e8, "FIG_DIR", tmp_path)
    orig = e8.fit
    df = common.load_dataset(labeled_only=True)
    t = df["time_step"].to_numpy()
    calls = []

    def spy_fit(X, y, w, seed):
        calls.append(X.copy())
        return orig(X, y, w, seed)

    monkeypatch.setattr(e8, "fit", spy_fit)
    a = e8.part_a(df, common.feature_cols(df, "time"), [0])
    Xall = df[common.feature_cols(df, "time")].to_numpy(np.float32)
    test_rows = {r.tobytes() for r in Xall[t >= 46]}
    for X in calls:
        assert not any(r.tobytes() in test_rows for r in X), "part A trained on a test row (steps 46-49)"
    assert set(a.condition) >= {"train_1_34", "train_1_42", "train_43_45", "train_1_42_post1_w10"}

    calls.clear()
    b = e8.part_b(df, common.feature_cols(df, "time"), [0])
    assert set(b.test_step) == set(range(35, 50))


def test_e8_main_writes_outputs(tmp_path, monkeypatch):
    make_synthetic_dataset(tmp_path / "data")
    monkeypatch.setattr(common, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(common, "RESULTS_PATH", tmp_path / "results.csv")
    monkeypatch.setattr(e8, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(e8, "FIG_DIR", tmp_path)
    assert e8.main(["--seeds", "0"]) == 0
    for f in ["e8_partA.csv", "e8_partB.csv", "e8_summary.csv", "e8_recovery.png", "e8_rolling.png"]:
        assert (tmp_path / f).exists(), f
    res = pd.read_csv(tmp_path / "results.csv", encoding="utf-8-sig")
    assert (res.experiment == "E8").sum() == len(pd.read_csv(tmp_path / "e8_partA.csv"))
