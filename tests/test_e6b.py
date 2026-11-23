"""Tests for E6b improved unsupervised learning experiment."""
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
import detectors_e6b
import e6b_improved
import prep


def make_synthetic_dataset(data_dir: Path):
    """Generate minimal synthetic elliptic CSVs with 165 features."""
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(42)

    rows = []
    classes = []
    tx_id = 2000

    # Steps 1..30 (train), 31..34 (val), 35..42 (test early), 43..49 (test late)
    for step in range(1, 50):
        # 4 rows per step for train, 5 for val, 4 for test
        n_step = 5 if 31 <= step <= 34 else 4
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


def test_prep_transformers_fit_only_on_fit_rows():
    """Assert a transformer fitted on train ignores test rows' values."""
    rng = np.random.RandomState(123)
    train_data = rng.randn(60, 166)
    test_data = rng.randn(30, 166)

    train_df = pd.DataFrame(train_data)
    train_df.columns = ["time_step"] + prep.ALL_FEATURES
    train_df["time_step"] = rng.choice(list(range(1, 31)), size=60)

    test_df = pd.DataFrame(test_data)
    test_df.columns = ["time_step"] + prep.ALL_FEATURES
    test_df["time_step"] = rng.choice(list(range(35, 50)), size=30)

    for p_name in prep.PREP_NAMES:
        t = prep.make_preprocessor(p_name, seed=0)
        t.fit(train_df)

        # Baseline transform of test_df
        out_base = t.transform(test_df)

        # Now heavily perturb test_df (should NOT affect t's internal fitted state)
        perturbed_test_df = test_df.copy()
        for col in prep.ALL_FEATURES:
            perturbed_test_df[col] = perturbed_test_df[col] * 1000.0 + 50000.0

        # Create another transformer fitted purely on train_df
        t2 = prep.make_preprocessor(p_name, seed=0)
        t2.fit(train_df)

        # The transformation of original test_df under t and t2 must be identical
        out_check = t2.transform(test_df)
        assert np.allclose(out_base, out_check, atol=1e-5), f"Fitted parameters varied for prep {p_name}"


def test_time_proxy_removed_in_p1_to_p3():
    """Requirement: TIME_PROXY removed in P1-P3."""
    expected_tp = ["agg_7", "agg_9", "agg_6", "agg_42", "agg_45", "agg_43"]
    assert sorted(prep.TIME_PROXY) == sorted(expected_tp)

    # P0 has all 165
    assert len(prep.get_feature_names("std")) == 165

    # P1, P2, P3 have 165 - 6 = 159
    for p_name in ["std_notp", "robust_clip", "quantile"]:
        fnames = prep.get_feature_names(p_name)
        assert len(fnames) == 159, f"Expected 159 features for {p_name}, got {len(fnames)}"
        for tp in expected_tp:
            assert tp not in fnames, f"{tp} unexpectedly found in {p_name}"

    # P4 has 93 local features
    p4_feats = prep.get_feature_names("stepnorm_local")
    assert len(p4_feats) == 93
    assert all(f.startswith("local_") for f in p4_feats)


def test_p4_uses_only_same_step_rows():
    """Requirement: P4 stepnorm_local uses only same-step rows."""
    rng = np.random.RandomState(456)
    rows = []
    # Create rows for step 35 and step 36
    for step in [35, 36]:
        for _ in range(10):
            feats = rng.randn(93)
            rows.append([step] + feats.tolist())

    df = pd.DataFrame(rows)
    df.columns = ["time_step"] + [f"local_{i}" for i in range(93)]

    # Fit standardizer on train-like data
    train_rows = []
    for step in range(1, 10):
        for _ in range(5):
            train_rows.append([step] + rng.randn(93).tolist())
    train_df = pd.DataFrame(train_rows)
    train_df.columns = ["time_step"] + [f"local_{i}" for i in range(93)]

    p4 = prep.StepwiseLocalStandardizer().fit(train_df)

    out_both = p4.transform(df)
    mask_35 = (df["time_step"] == 35).values
    out_35_from_both = out_both[mask_35]

    # Transform step 35 only
    df_step35_only = df[mask_35].copy()
    out_35_alone = p4.transform(df_step35_only)

    assert np.allclose(out_35_from_both, out_35_alone, atol=1e-5), (
        "P4 transform of step 35 differed when step 36 was present"
    )

    # Now mutate step 36 radically in the combined dataframe
    df_mutated = df.copy()
    mask_36 = (df_mutated["time_step"] == 36).values
    for col in [f"local_{i}" for i in range(93)]:
        df_mutated.loc[mask_36, col] = df_mutated.loc[mask_36, col] * 1000.0 + 9999.0

    out_mutated = p4.transform(df_mutated)
    out_35_after_mutated = out_mutated[mask_35]

    assert np.allclose(out_35_alone, out_35_after_mutated, atol=1e-5), (
        "Mutating step 36 influenced the normalized output of step 35 in P4!"
    )


def test_gmm_component_denser_component_lower_score():
    """Requirement: gmm_component scorer picks denser component as lower score."""
    rng = np.random.RandomState(789)
    # Cluster 0: dense (small variance)
    cov0 = np.eye(4) * 0.05
    # Cluster 1: dispersed (large variance)
    cov1 = np.eye(4) * 10.0

    x0 = rng.multivariate_normal(np.zeros(4), cov0, size=150).astype(np.float32)
    x1 = rng.multivariate_normal(np.ones(4) * 15.0, cov1, size=150).astype(np.float32)
    X = np.vstack([x0, x1])

    det = detectors_e6b.GMMDetector(n_components=2, pca=None, seed=0).fit(X)
    scores = det.score_component(X)

    score_dense = scores[:150]
    score_sparse = scores[150:]

    mean_dense = np.mean(score_dense)
    mean_sparse = np.mean(score_sparse)

    assert mean_dense < mean_sparse, (
        f"Expected denser component score ({mean_dense:.3f}) < sparse component score ({mean_sparse:.3f})"
    )


def test_e6b_fast_synthetic(tmp_path, monkeypatch):
    """Requirement: --fast run on synthetic data writes all outputs and E6b rows."""
    data_dir = tmp_path / "data" / "elliptic"
    results_dir = tmp_path / "results"
    fig_dir = tmp_path / "report" / "figures"
    model_dir = tmp_path / "model"

    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    make_synthetic_dataset(data_dir)

    results_csv = results_dir / "results.csv"

    # Monkeypatch paths
    monkeypatch.setattr(common, "DATA_DIR", data_dir)
    monkeypatch.setattr(common, "RESULTS_PATH", results_csv)
    monkeypatch.setattr(e6b_improved, "FIG_DIR", fig_dir)
    monkeypatch.setattr(e6b_improved, "MODEL_DIR", model_dir)
    monkeypatch.setattr(e6b_improved, "RESULTS_DIR", results_dir)

    # Run e6b_improved with --fast
    ret = e6b_improved.main(["--fast"])
    assert ret == 0

    # 1. Check results files
    expected_result_files = ["e6b_grid.csv", "e6b_summary.csv", "e6b_prep_compare.csv", "e6b_by_step.csv"]
    for fname in expected_result_files:
        p = results_dir / fname
        assert p.exists(), f"Missing result file: {p}"
        assert p.stat().st_size > 0, f"Empty result file: {p}"

    # 2. Check 4 figures
    expected_figures = [
        "e6b_prep_compare.png",
        "e6b_test_pr_auc.png",
        "e6b_by_step.png",
        "e6b_gmm_components.png",
    ]
    for fname in expected_figures:
        p = fig_dir / fname
        assert p.exists(), f"Missing figure: {p}"
        assert p.stat().st_size > 0, f"Empty figure: {p}"

    # 3. Check 8 dumped models (one per model group in mode best) + ref npy + manifest
    for g in e6b_improved.MODEL_GROUPS:
        m_file = model_dir / f"e6b_{g}.joblib"
        r_file = model_dir / f"e6b_{g}_ref.npy"
        assert m_file.exists(), f"Missing dumped model: {m_file}"
        assert r_file.exists(), f"Missing dumped ref: {r_file}"

    manifest_p = model_dir / "e6b_manifest.json"
    assert manifest_p.exists(), "Missing manifest.json"
    manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
    assert manifest["experiment"] == "E6b"
    assert manifest["seeds"] == [0, 1, 2]
    assert len(manifest["models"]) == 8

    # 4. Check results.csv for E6b entries
    assert results_csv.exists(), "Missing results/results.csv"
    res_df = pd.read_csv(results_csv)
    e6b_rows = res_df[res_df["experiment"] == "E6b"]
    assert len(e6b_rows) > 0, "No E6b rows found in results.csv"
    assert set(e6b_rows["seed"].unique()) == {0, 1, 2}

    # Verify model label format f"{Model}[{prep}]({mode})"
    for m in e6b_rows["model"].unique():
        assert "[" in m and "]" in m and "(" in m and ")" in m, f"Model label {m} does not match format"

    # Verify feature_set values
    assert set(e6b_rows["feature_set"].unique()).issubset({"all", "all-tp", "local"})

    # Verify note contents
    for _, row in e6b_rows.iterrows():
        note = str(row["note"])
        assert "params=" in note
        assert "direction=" in note
        assert "pr_auc_35_42=" in note
        assert "pr_auc_43_49=" in note
        assert "threshold=tuned_on_val" in note

    # 5. Check e6b_summary.csv has no duplicated column sets
    summary_df = pd.read_csv(results_dir / "e6b_summary.csv")
    cols = list(summary_df.columns)
    assert len(cols) == len(set(cols)), "Duplicate columns in e6b_summary.csv"
    assert "precision_mean" in cols and "test_precision_mean" not in cols
    assert "pr_auc_mean" in cols and "test_pr_auc_mean" not in cols


def test_fit_rows_grid_stage_no_future(tmp_path, monkeypatch):
    """Requirement: fit rows never include time_step > 30 in grid stage."""
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
    monkeypatch.setattr(e6b_improved, "FIG_DIR", fig_dir)
    monkeypatch.setattr(e6b_improved, "MODEL_DIR", model_dir)
    monkeypatch.setattr(e6b_improved, "RESULTS_DIR", results_dir)

    ret = e6b_improved.main(["--fast"])
    assert ret == 0

    assert hasattr(e6b_improved, "GRID_FIT_MAX_STEP")
    assert e6b_improved.GRID_FIT_MAX_STEP <= 30

    grid_df = pd.read_csv(results_dir / "e6b_grid.csv")
    assert "fit_max_t" in grid_df.columns
    assert (grid_df["fit_max_t"] <= 30).all(), f"Found fit_max_t > 30: {grid_df['fit_max_t'].max()}"


def test_stepnorm_uses_all_rows_of_step_not_the_batch():
    """P4 must normalize with stats of ALL rows of the step (incl. unknown), not of the passed subset."""
    import numpy as np, pandas as pd, sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import prep
    rng = np.random.RandomState(0)
    n = 400
    df = pd.DataFrame(rng.normal(size=(n, 93)), columns=[f"local_{i}" for i in range(93)])
    df["time_step"] = np.repeat([1, 2], n // 2)
    df["y"] = np.where(rng.rand(n) < 0.3, 1.0, np.nan)
    prep.set_step_stats(df)
    p = prep.make_preprocessor("stepnorm_local", seed=0).fit(df)
    full = p.transform(df)
    sub_idx = np.flatnonzero(df["y"].notna().values)
    sub = p.transform(df.iloc[sub_idx])
    np.testing.assert_allclose(sub, full[sub_idx], rtol=1e-5, atol=1e-5)


def test_birch_detector_pickles_with_many_subclusters(tmp_path):
    """BIRCH CF-tree with thousands of leaves hit RecursionError in joblib.dump on Colab."""
    import sys, joblib, numpy as np
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import detectors_e6b as D
    X = np.random.RandomState(0).standard_t(2, size=(10000, 40)).astype("float32")
    det = D.make_detector("birch", {"threshold": 0.5, "n_clusters": 8, "pca": 20}, seed=0).fit(X)
    before = det.score(X[:100])
    joblib.dump(det, tmp_path / "b.joblib")
    after = joblib.load(tmp_path / "b.joblib").score(X[:100])
    np.testing.assert_allclose(before, after)


@pytest.fixture
def setting_b():
    prep.configure("B")
    yield
    prep.configure("A")


def test_setting_b_uses_only_transaction_time_features(setting_b):
    """Setting B: 57 features (TX local_0-20 + NBR_IN agg_0-35); no address stats, no successor aggregates."""
    tx = [f"local_{i}" for i in range(21)]
    nbr_in = [f"agg_{i}" for i in range(36)]
    assert prep.get_feature_names("std") == tx + nbr_in
    assert prep.get_feature_names("quantile_all") == tx + nbr_in
    assert sorted(prep.TIME_PROXY) == ["agg_6", "agg_7", "agg_9"]
    for p_name in ["std_notp", "robust_clip", "quantile"]:
        f = prep.get_feature_names(p_name)
        assert len(f) == 54 and not set(f) & {"agg_6", "agg_7", "agg_9"}
    assert prep.get_feature_names("stepnorm_local") == tx
    for p_name in prep.PREP_NAMES:
        assert set(prep.get_feature_names(p_name)) <= set(tx + nbr_in), p_name


def test_e6b_fast_synthetic_setting_b(tmp_path, monkeypatch, setting_b):
    """--setting B writes e6b_B_* outputs, logs E6b-B rows, leaves E6b rows alone."""
    data_dir = tmp_path / "data" / "elliptic"
    results_dir, fig_dir, model_dir = tmp_path / "results", tmp_path / "fig", tmp_path / "model"
    for d in (results_dir, fig_dir, model_dir):
        d.mkdir(parents=True, exist_ok=True)
    make_synthetic_dataset(data_dir)
    results_csv = results_dir / "results.csv"
    pd.DataFrame([{"timestamp": "x", "experiment": "E6b", "model": "keep", "feature_set": "all", "seed": 0,
                   "precision": 0, "recall": 0, "f1": 0, "pr_auc": 0, "threshold": 0.5, "note": ""}]
                 ).to_csv(results_csv, index=False, encoding="utf-8-sig")
    monkeypatch.setattr(common, "DATA_DIR", data_dir)
    monkeypatch.setattr(common, "RESULTS_PATH", results_csv)
    monkeypatch.setattr(e6b_improved, "FIG_DIR", fig_dir)
    monkeypatch.setattr(e6b_improved, "MODEL_DIR", model_dir)
    monkeypatch.setattr(e6b_improved, "RESULTS_DIR", results_dir)

    assert e6b_improved.main(["--fast", "--setting", "B"]) == 0

    for fname in ["e6b_B_grid.csv", "e6b_B_summary.csv", "e6b_B_prep_compare.csv", "e6b_B_by_step.csv"]:
        assert (results_dir / fname).exists(), fname
    assert not (results_dir / "e6b_grid.csv").exists()
    assert (fig_dir / "e6b_B_test_pr_auc.png").exists()
    manifest = json.loads((model_dir / "e6b_B_manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment"] == "E6b-B"
    res = pd.read_csv(results_csv, encoding="utf-8-sig")
    assert (res["experiment"] == "E6b").sum() == 1  # setting A rows untouched
    b = res[res["experiment"] == "E6b-B"]
    assert len(b) > 0 and set(b["feature_set"]) <= {"time", "time-tp", "time-tx"}
    grid = pd.read_csv(results_dir / "e6b_B_grid.csv")
    assert "quantile_all" in set(grid["prep"])
