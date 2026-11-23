"""Tests for E8b (semi-supervised recovery after the shift)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common
import e8b_shift_ssl as e8b
from test_e6b import make_synthetic_dataset


def test_e8b_trains_only_on_steps_up_to_45(tmp_path, monkeypatch):
    make_synthetic_dataset(tmp_path / "data")
    monkeypatch.setattr(common, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(e8b, "R", tmp_path)
    monkeypatch.setattr(e8b, "F", tmp_path)
    df = common.load_dataset(labeled_only=False)
    X = df[common.feature_cols(df, "time")].to_numpy(np.float32)
    later = {r.tobytes() for r in X[df.time_step.to_numpy() >= 46]}
    seen = []
    orig_st = e8b.self_train

    def spy_st(XL, yL, XU, seed, **kw):
        seen.extend([XL, XU])
        return orig_st(XL, yL, XU, seed, **kw)

    monkeypatch.setattr(e8b, "self_train", spy_st)
    assert e8b.main(seeds=[0], post_fracs=[0.0, 0.5]) == 0
    for A in seen:
        assert not any(r.tobytes() in later for r in A), "trained on a test-period row (46-49)"
    runs = pd.read_csv(tmp_path / "e8b_runs.csv")
    assert set(runs.method) == {"LGBM", "ST", "LS", "ENS"}
    assert runs.pr_auc_46_49.notna().all()
    assert (runs[runs.post_frac == 0].n_post_labels == 0).all()
