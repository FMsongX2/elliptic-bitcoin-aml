"""Tests for E7d (causal test-time self-training)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common
import e7d_test_time_st as e7d
from test_e6b import make_synthetic_dataset


def test_causal_mode_never_uses_rows_after_scored_step(tmp_path, monkeypatch):
    make_synthetic_dataset(tmp_path / "data")
    monkeypatch.setattr(common, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(e7d, "R", tmp_path)
    monkeypatch.setattr(e7d, "F", tmp_path)
    monkeypatch.setattr(e7d.e7c, "MIN_LABELS", 10)
    df = common.load_dataset(labeled_only=False)
    X = df[common.feature_cols(df, "time")].to_numpy(np.float32)
    t = df["time_step"].to_numpy()
    seen = []
    orig = e7d.self_train

    def spy(XL, yL, XU, seed, **kw):
        seen.append(XU.copy())
        return orig(XL, yL, XU, seed, **kw)

    monkeypatch.setattr(e7d, "self_train", spy)
    assert e7d.main(["--fracs", "0.2", "--seeds", "0", "--steps", "36", "40"]) == 0
    later = {r.tobytes() for r in X[t > 40]}
    after36 = {r.tobytes() for r in X[t > 36]}
    # calls: static (once), causal@36, causal@40
    assert len(seen) == 3
    assert not any(r.tobytes() in after36 for r in seen[1])
    assert not any(r.tobytes() in later for r in seen[2])
    assert any(r.tobytes() in {x.tobytes() for x in X[t == 40]} for r in seen[2])
    runs = pd.read_csv(tmp_path / "e7d_runs.csv")
    assert set(runs["mode"]) == {"static", "causal"} and set(runs.step) == {36, 40}
    assert (tmp_path / "e7d_summary.csv").exists()
