"""E6b Improved Unsupervised Experiment.

Team protocol matching e6_unsupervised.py:
- Grid fit on ALL rows (incl. unknown) time_step 1-30 (TRAIN_STEPS) without labels.
- Selection of config + score direction (+1/-1) + threshold on labeled VAL_STEPS (31-34)
  via val PR-AUC and tune_threshold on ECDF probabilities (to_prob).
- Final refit on 1-34 with seeds [0, 1, 2], evaluate on labeled TEST_STEPS (35-49).
- Log results to results/results.csv via log_result(EXP, ...).
- Resumable grid results/e6b_grid.csv.
- --fast mode for smoke tests.

Preprocessing options (P0-P4 from prep.py):
- P0 "std": all 165 features, StandardScaler.
- P1 "std_notp": 165 minus TIME_PROXY, StandardScaler.
- P2 "robust_clip": 165 minus TIME_PROXY, clip to 0.5/99.5 percentiles, RobustScaler.
- P3 "quantile": 165 minus TIME_PROXY, QuantileTransformer(output_distribution="normal").
- P4 "stepnorm_local": local_* only (93), per-time-step standardized + StandardScaler.

Model groups:
a) GMM (full covariance):
   - gmm_density: -score_samples
   - gmm_component: 0.5 * logdet(Sigma_c) (denser component = lower score)
b) HDBSCAN on PCA {10, 20}: nearest clustered point dist * (1 + label_noise)
c) Isolation Forest (n_estimators 300, max_samples {1024,4096}, max_features {0.5,1.0})
d) PCA reconstruction error: squared reconstruction error
e) Kernel PCA reconstruction error: rbf, gamma {0.01,0.05}, 5k subsample
f) One-class SVM on PCA-20: -decision_function, 10k subsample
g) BIRCH on PCA-20: distance to nearest subcluster centroid
"""
import gc
import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, f1_score

import common
from common import (
    DARK_MARKET_STEP,
    TEST_STEPS,
    TRAIN_STEPS,
    VAL_STEPS,
    clear_results,
    evaluate,
    evaluate_by_step,
    load_dataset,
    log_result,
    tune_threshold,
)
from detectors_e6b import (
    BirchDetector,
    GMMDetector,
    HDBSCANDetector,
    IForestDetector,
    KPCAReconDetector,
    OCSVMDetector,
    PCAReconDetector,
    make_detector,
)
from e6_unsupervised import to_prob
from common import BASE
import prep

# Configurable directories for tests
FIG_DIR = BASE / "figures"
MODEL_DIR = BASE / "models"
RESULTS_DIR = BASE / "results"

GRID_FIT_MAX_STEP = 30

# Output naming per evaluation setting (set in main): A -> "e6b_*" / "E6b", B -> "e6b_B_*" / "E6b-B"
TAG = "e6b"
EXP = "E6b"

MODEL_GROUPS = [
    "pca_recon",
    "iforest",
    "gmm_density",
    "gmm_component",
    "birch",
    "ocsvm",
    "kpca_recon",
    "hdbscan",
]

MODEL_LABELS = {
    "pca_recon": "PCA-recon",
    "iforest": "IForest",
    "gmm_density": "GMM-density",
    "gmm_component": "GMM-component",
    "birch": "BIRCH",
    "ocsvm": "OCSVM",
    "kpca_recon": "KPCA-recon",
    "hdbscan": "HDBSCAN",
}


def _json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _append_row(path: Path, fieldnames: List[str], row: Dict[str, Any]):
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new:
            w.writeheader()
        w.writerow(row)


def get_feature_set_name(prep_name: str) -> str:
    """Return feature_set string for team results logging."""
    if prep.SETTING == "B":
        if prep_name == "stepnorm_local":
            return "time-tx"
        if prep_name in ("std", "quantile_all"):
            return "time"
        return "time-tp"
    if prep_name == "stepnorm_local":
        return "local"
    if prep_name == "std":
        return "all"
    return "all-tp"


def _make_grid_specs_stage1(fast: bool = False):
    """Generate (model_type, params) for Stage 1 (PCA recon, IForest, GMM)."""
    specs = []
    # (d) PCA recon: cheap, runs first
    pca_comps = [5] if fast else [5, 10, 20]
    for k in pca_comps:
        specs.append(("pca_recon", {"n_components": k}))

    # (c) Isolation Forest
    if fast:
        specs.append(("iforest", {"n_estimators": 50, "max_samples": 256, "max_features": 0.5}))
    else:
        for s in (1024, 4096):
            for f in (0.5, 1.0):
                specs.append(("iforest", {"n_estimators": 300, "max_samples": s, "max_features": f}))

    # (a) GMM
    if fast:
        specs.append(("gmm", {"n_components": 8, "pca": 20}))
    else:
        for k in (8, 12, 16, 20):
            for p in (20, 0.95):
                specs.append(("gmm", {"n_components": k, "pca": p}))

    return specs


def _make_grid_specs_stage2(fast: bool = False):
    """Generate (model_type, params) for Stage 2 (BIRCH, OCSVM, KPCA, HDBSCAN)."""
    specs = []
    # (g) BIRCH
    if fast:
        specs.append(("birch", {"threshold": 0.5, "n_clusters": 8, "pca": 20}))
    else:
        for th in (0.5, 1.0):
            for nc in (8, 16):
                specs.append(("birch", {"threshold": th, "n_clusters": nc, "pca": 20}))

    # (f) OCSVM
    if fast:
        specs.append(("ocsvm", {"nu": 0.05, "gamma": "scale", "pca": 20}))
    else:
        for nu in (0.05, 0.2):
            for gam in ("scale", 0.01):
                specs.append(("ocsvm", {"nu": nu, "gamma": gam, "pca": 20}))

    # (e) KPCA recon
    if fast:
        specs.append(("kpca_recon", {"n_components": 5, "gamma": 0.01}))
    else:
        for gam in (0.01, 0.05):
            specs.append(("kpca_recon", {"n_components": 20, "gamma": gam}))

    # (b) HDBSCAN
    if fast:
        specs.append(("hdbscan", {"pca": 10, "min_cluster_size": 25, "min_samples": 5}))
    else:
        for p in (10, 20):
            for mcs in (25, 50, 100, 200):
                for ms in (5, 10, 20):
                    specs.append(("hdbscan", {"pca": p, "min_cluster_size": mcs, "min_samples": ms}))

    return specs


def main(argv=None):
    global GRID_FIT_MAX_STEP, TAG, EXP
    parser = argparse.ArgumentParser(description="E6b Improved Unsupervised experiment on team conventions.")
    parser.add_argument("--fast", action="store_true", help="Run fast smoke test (subsample splits and configs)")
    parser.add_argument("--setting", choices=["A", "B"], default="A",
                        help="A = all 165 features, B = 57 transaction-time features (TX + NBR_IN)")
    args = parser.parse_args(argv)
    prep.configure(args.setting)
    TAG, EXP = ("e6b", "E6b") if args.setting == "A" else ("e6b_B", "E6b-B")
    print(f"Setting {args.setting}: {len(prep.ALL_FEATURES)} features, time proxy {prep.TIME_PROXY}", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Elliptic dataset...", flush=True)
    df = load_dataset(labeled_only=False)
    prep.set_step_stats(df)  # P4: per-step stats over all rows of each step (unlabeled features only)
    print(f"Loaded {len(df)} transactions across 49 steps.", flush=True)

    t_all = df["time_step"].values
    y_all = df["y"].values

    # Splits
    train_mask = np.isin(t_all, list(TRAIN_STEPS))
    val_mask = np.isin(t_all, list(VAL_STEPS)) & pd.notna(y_all)
    test_mask = np.isin(t_all, list(TEST_STEPS)) & pd.notna(y_all)
    full_mask = np.isin(t_all, list(TRAIN_STEPS) + list(VAL_STEPS))

    if args.fast:
        rng = np.random.RandomState(0)
        for mask, max_n in [(train_mask, 2000), (val_mask, 2000), (test_mask, 2000), (full_mask, 2000)]:
            idxs = np.flatnonzero(mask)
            if len(idxs) > max_n:
                mask[:] = False
                mask[rng.choice(idxs, max_n, replace=False)] = True

    GRID_FIT_MAX_STEP = int(t_all[train_mask].max())
    print(
        f"Splits: fit_train={train_mask.sum()} (max t={GRID_FIT_MAX_STEP}), "
        f"val={val_mask.sum()}, test={test_mask.sum()}, fit_full={full_mask.sum()}",
        flush=True,
    )

    df_train = df[train_mask].copy()
    df_val = df[val_mask].copy()
    df_test = df[test_mask].copy()
    df_full = df[full_mask].copy()

    y_val = df_val["y"].values.astype(int)
    y_test = df_test["y"].values.astype(int)

    # -------------------------------------------------------------
    # Grid stage (resumable)
    # -------------------------------------------------------------
    grid_csv = RESULTS_DIR / f"{TAG}_grid.csv"
    grid_fields = [
        "config_id",
        "model",
        "prep",
        "params",
        "val_pr_auc",
        "val_pr_auc_neg",
        "fit_seconds",
        "fit_max_t",
        "val_scores_json",
        "extra_json",
    ]
    existing = {}
    if grid_csv.exists() and grid_csv.stat().st_size > 0:
        with grid_csv.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                existing[r["config_id"]] = r

    # Cache preprocessed features for train and val to avoid redundant transforms
    print("Fitting preprocessors on train rows (1-30)...", flush=True)
    prep_data_train = {}
    prep_data_val = {}
    prep_transformers_train = {}
    for p_name in prep.PREP_NAMES:
        p_trans = prep.make_preprocessor(p_name, seed=0).fit(df_train)
        prep_transformers_train[p_name] = p_trans
        prep_data_train[p_name] = p_trans.transform(df_train)
        prep_data_val[p_name] = p_trans.transform(df_val)

    records = []

    def _eval_and_record(group_name, prep_name, params, det, score_val, fit_sec, extras=None):
        config_id = f"{group_name}:{prep_name}:{_json(params)}"
        if config_id in existing:
            r = existing[config_id]
            val_scores = np.asarray(json.loads(r["val_scores_json"]), dtype=np.float32)
            pr_auc = float(r["val_pr_auc"])
            pr_auc_neg = float(r.get("val_pr_auc_neg", average_precision_score(y_val, -val_scores)))
            extra_d = json.loads(r.get("extra_json", "{}"))
            records.append({
                "config_id": config_id,
                "model": group_name,
                "prep": prep_name,
                "params": params,
                "val_scores": val_scores,
                "val_pr_auc": pr_auc,
                "val_pr_auc_neg": pr_auc_neg,
                "fit_seconds": float(r["fit_seconds"]),
                "detector": None,
                "extra": extra_d,
            })
            print(f"[{group_name}][{prep_name}] {_json(params)} val_pr_auc={pr_auc:.4f} (cached)", flush=True)
            return pr_auc, pr_auc_neg

        val_scores = np.asarray(score_val, dtype=np.float32)
        pr_auc = float(average_precision_score(y_val, val_scores))
        pr_auc_neg = float(average_precision_score(y_val, -val_scores))
        extra_d = extras or {}

        row = {
            "config_id": config_id,
            "model": group_name,
            "prep": prep_name,
            "params": _json(params),
            "val_pr_auc": round(pr_auc, 5),
            "val_pr_auc_neg": round(pr_auc_neg, 5),
            "fit_seconds": round(fit_sec, 3),
            "fit_max_t": GRID_FIT_MAX_STEP,
            "val_scores_json": _json(val_scores.tolist()),
            "extra_json": _json(extra_d),
        }
        _append_row(grid_csv, grid_fields, row)
        records.append({
            "config_id": config_id,
            "model": group_name,
            "prep": prep_name,
            "params": params,
            "val_scores": val_scores,
            "val_pr_auc": pr_auc,
            "val_pr_auc_neg": pr_auc_neg,
            "fit_seconds": fit_sec,
            "detector": None,  # do not keep fitted models in memory (Colab OOM); refit selected ones later
            "extra": extra_d,
        })
        gc.collect()
        print(f"[{group_name}][{prep_name}] {_json(params)} val_pr_auc={pr_auc:.4f} ({fit_sec:.2f}s)", flush=True)
        return pr_auc, pr_auc_neg

    # -------------------------------------------------------------
    # Stage 1: P0-P4 for (a) GMM, (c) IForest, (d) PCA recon
    # -------------------------------------------------------------
    specs_stage1 = _make_grid_specs_stage1(fast=args.fast)
    print(f"\n--- Running Stage 1 on P0-P4 ({len(specs_stage1) * len(prep.PREP_NAMES)} fits) ---", flush=True)

    for p_name in prep.PREP_NAMES:
        X_tr = prep_data_train[p_name]
        X_v = prep_data_val[p_name]

        for model_type, params in specs_stage1:
            if model_type == "gmm":
                cid_density = f"gmm_density:{p_name}:{_json(params)}"
                cid_comp = f"gmm_component:{p_name}:{_json(params)}"
                if cid_density in existing and cid_comp in existing:
                    _eval_and_record("gmm_density", p_name, params, None, None, 0.0)
                    _eval_and_record("gmm_component", p_name, params, None, None, 0.0)
                    continue

                t0 = time.perf_counter()
                gmm_det = GMMDetector(seed=0, **params).fit(X_tr)
                fit_sec = time.perf_counter() - t0
                s_dens = gmm_det.score_density(X_v)
                s_comp = gmm_det.score_component(X_v)

                _eval_and_record("gmm_density", p_name, params, gmm_det, s_dens, fit_sec)
                _eval_and_record("gmm_component", p_name, params, gmm_det, s_comp, fit_sec)
            else:
                cid = f"{model_type}:{p_name}:{_json(params)}"
                if cid in existing:
                    _eval_and_record(model_type, p_name, params, None, None, 0.0)
                    continue

                t0 = time.perf_counter()
                det = make_detector(model_type, params, seed=0).fit(X_tr)
                fit_sec = time.perf_counter() - t0
                s_val = det.score(X_v)
                _eval_and_record(model_type, p_name, params, det, s_val, fit_sec)

    # -------------------------------------------------------------
    # Decide P2 vs P3 using Isolation Forest val PR-AUC
    # -------------------------------------------------------------
    iforest_p2 = [r for r in records if r["model"] == "iforest" and r["prep"] == "robust_clip"]
    iforest_p3 = [r for r in records if r["model"] == "iforest" and r["prep"] == "quantile"]

    best_p2_score = max([max(r["val_pr_auc"], r["val_pr_auc_neg"]) for r in iforest_p2]) if iforest_p2 else 0.0
    best_p3_score = max([max(r["val_pr_auc"], r["val_pr_auc_neg"]) for r in iforest_p3]) if iforest_p3 else 0.0

    best_p2_p3 = "robust_clip" if best_p2_score >= best_p3_score else "quantile"
    print(
        f"\nIForest comparison: P2 (robust_clip)={best_p2_score:.4f} vs P3 (quantile)={best_p3_score:.4f}. "
        f"Selected best for Stage 2: {best_p2_p3}",
        flush=True,
    )

    # -------------------------------------------------------------
    # Stage 2: P1 and best_p2_p3 for (b) HDBSCAN, (e) KPCA, (f) OCSVM, (g) BIRCH
    # -------------------------------------------------------------
    stage2_preps = ["std_notp", best_p2_p3]
    if "quantile" not in stage2_preps:  # always cover the setting-A winner (BIRCH + quantile) for comparison
        stage2_preps.append("quantile")
    specs_stage2 = _make_grid_specs_stage2(fast=args.fast)
    print(f"\n--- Running Stage 2 on {stage2_preps} ({len(specs_stage2) * len(stage2_preps)} fits) ---", flush=True)

    for p_name in stage2_preps:
        X_tr = prep_data_train[p_name]
        X_v = prep_data_val[p_name]

        for model_type, params in specs_stage2:
            cid = f"{model_type}:{p_name}:{_json(params)}"
            if cid in existing:
                _eval_and_record(model_type, p_name, params, None, None, 0.0)
                continue

            t0 = time.perf_counter()
            det = make_detector(model_type, params, seed=0).fit(X_tr)
            fit_sec = time.perf_counter() - t0
            s_val = det.score(X_v)
            extras = {}
            if model_type == "hdbscan" and hasattr(det, "cluster_sizes_"):
                extras["cluster_sizes"] = det.cluster_sizes_
            _eval_and_record(model_type, p_name, params, det, s_val, fit_sec, extras=extras)

    # -------------------------------------------------------------
    # Model Selection per (mode x model group x prep)
    # -------------------------------------------------------------
    print("\nSelecting best configs on validation set...", flush=True)
    selected_configs = {}  # key: (mode, group, prep) -> {"record": r, "direction": dir}

    for group in MODEL_GROUPS:
        # Preps evaluated for this group
        group_records = [r for r in records if r["model"] == group]
        preps_evaluated = sorted(list(set(r["prep"] for r in group_records)))

        for p_name in preps_evaluated:
            candidates = [r for r in group_records if r["prep"] == p_name]
            if not candidates:
                continue

            # Anomaly mode: direction +1 only
            best_anom = max(candidates, key=lambda r: r["val_pr_auc"])
            selected_configs[("anomaly", group, p_name)] = {"record": best_anom, "direction": 1}

            # Best mode: direction +1 or -1
            best_rec, best_dir, best_score = None, 1, -1.0
            for r in candidates:
                if r["val_pr_auc"] > best_score:
                    best_score = r["val_pr_auc"]
                    best_rec = r
                    best_dir = 1
                if r["val_pr_auc_neg"] > best_score:
                    best_score = r["val_pr_auc_neg"]
                    best_rec = r
                    best_dir = -1
            selected_configs[("best", group, p_name)] = {"record": best_rec, "direction": best_dir}

    # Prep compare summary (best val PR-AUC per model group per prep in mode "best")
    prep_compare_rows = []
    for group in MODEL_GROUPS:
        row = {"model": MODEL_LABELS[group]}
        for p_name in prep.PREP_NAMES:
            key = ("best", group, p_name)
            if key in selected_configs:
                rec = selected_configs[key]["record"]
                d = selected_configs[key]["direction"]
                val_auc = rec["val_pr_auc"] if d == 1 else rec["val_pr_auc_neg"]
                row[p_name] = round(val_auc, 4)
            else:
                row[p_name] = np.nan
        prep_compare_rows.append(row)
    prep_compare_df = pd.DataFrame(prep_compare_rows)
    prep_compare_df.to_csv(RESULTS_DIR / f"{TAG}_prep_compare.csv", index=False, encoding="utf-8-sig")
    print(f"Saved results/{TAG}_prep_compare.csv ({len(prep_compare_df)} rows).", flush=True)

    # Keep the final (refit x 3 seeds) stage affordable: per (mode, group) keep only the prep with the
    # best val PR-AUC, plus P0 "std" and P1 "std_notp" in mode "best" to measure the time-proxy effect
    # on test. The full prep comparison is reported on val in e6b_prep_compare.csv.
    def _val_of(choice):
        r = choice["record"]
        return r["val_pr_auc"] if choice["direction"] == 1 else r["val_pr_auc_neg"]

    kept = {}
    for (mode, group, p_name), choice in selected_configs.items():
        best_key = max((k for k in selected_configs if k[0] == mode and k[1] == group),
                       key=lambda k: _val_of(selected_configs[k]))
        if (mode, group, p_name) == best_key or (mode == "best" and p_name in ("std", "std_notp")):
            kept[(mode, group, p_name)] = choice
    print(f"Final stage keeps {len(kept)} of {len(selected_configs)} (mode, group, prep) configs.", flush=True)
    selected_configs = kept

    # Threshold tuning on validation set
    thresholds = {}
    for key, choice in selected_configs.items():
        mode, group, p_name = key
        rec = choice["record"]
        direction = choice["direction"]

        det = rec["detector"]
        X_tr = prep_data_train[p_name]
        if det is None:
            if group in ("gmm_density", "gmm_component"):
                det = GMMDetector(seed=0, **rec["params"]).fit(X_tr)
            else:
                det = make_detector(group, rec["params"], seed=0).fit(X_tr)

        if group == "gmm_density":
            fit_scores = direction * det.score_density(X_tr)
        elif group == "gmm_component":
            fit_scores = direction * det.score_component(X_tr)
        else:
            fit_scores = direction * det.score(X_tr)

        val_scores = direction * rec["val_scores"]
        prob_val = to_prob(fit_scores, val_scores)
        thresholds[key] = tune_threshold(y_val, prob_val)

    # -------------------------------------------------------------
    # Final Stage: refit on fit_full (steps 1-34) with seeds [0, 1, 2]
    # -------------------------------------------------------------
    print("\nStarting final stage: refitting on fit_full (steps 1-34) with seeds [0, 1, 2]...", flush=True)
    clear_results(EXP)

    early_mask = (df_test["time_step"] <= 42).values
    late_mask = (df_test["time_step"] >= 43).values

    summary_records = []
    seed0_models = {}
    seed0_test_probs = {}
    manifest_models = {}

    # Pre-cache fit_full preprocessors for seeds [0, 1, 2]
    prep_data_full = {}
    prep_data_test = {}
    # Only preps used by the kept configs; only "quantile" depends on the seed (subsampling),
    # the others are deterministic, so they are computed once and shared across seeds (saves memory).
    for p_name in sorted({k[2] for k in selected_configs}):
        for seed in ([0, 1, 2] if p_name == "quantile" else [0]):
            p_trans = prep.make_preprocessor(p_name, seed=seed).fit(df_full)
            prep_data_full[(p_name, seed)] = p_trans.transform(df_full)
            prep_data_test[(p_name, seed)] = p_trans.transform(df_test)
        if p_name != "quantile":
            for seed in (1, 2):
                prep_data_full[(p_name, seed)] = prep_data_full[(p_name, 0)]
                prep_data_test[(p_name, seed)] = prep_data_test[(p_name, 0)]
        gc.collect()

    # Identify overall best prep per group for mode "best" (for dumped models & step evaluation)
    overall_best_for_group = {}
    for group in MODEL_GROUPS:
        best_p, best_val_auc = None, -1.0
        for p_name in prep.PREP_NAMES:
            key = ("best", group, p_name)
            if key in selected_configs:
                rec = selected_configs[key]["record"]
                d = selected_configs[key]["direction"]
                auc = rec["val_pr_auc"] if d == 1 else rec["val_pr_auc_neg"]
                if auc > best_val_auc:
                    best_val_auc = auc
                    best_p = p_name
        overall_best_for_group[group] = best_p

    for key, choice in selected_configs.items():
        mode, group, p_name = key
        rec = choice["record"]
        direction = choice["direction"]
        threshold = thresholds[key]
        val_pr_auc = rec["val_pr_auc"] if direction == 1 else rec["val_pr_auc_neg"]

        precisions, recalls, f1s, pr_aucs = [], [], [], []
        f1_earlies, pr_auc_earlies = [], []
        f1_lates, pr_auc_lates = [], []

        for seed in [0, 1, 2]:
            X_fu = prep_data_full[(p_name, seed)]
            X_te = prep_data_test[(p_name, seed)]

            if group in ("gmm_density", "gmm_component"):
                det = GMMDetector(seed=seed, **rec["params"]).fit(X_fu)
                if group == "gmm_density":
                    fit_scores = direction * det.score_density(X_fu)
                    test_scores = direction * det.score_density(X_te)
                else:
                    fit_scores = direction * det.score_component(X_fu)
                    test_scores = direction * det.score_component(X_te)
            else:
                det = make_detector(group, rec["params"], seed=seed).fit(X_fu)
                fit_scores = direction * det.score(X_fu)
                test_scores = direction * det.score(X_te)

            prob_test = to_prob(fit_scores, test_scores)
            m = evaluate(y_test, prob_test, threshold=threshold)

            precisions.append(m["precision"])
            recalls.append(m["recall"])
            f1s.append(m["f1"])
            pr_aucs.append(m["pr_auc"])

            # Early & late metrics
            if len(np.unique(y_test[early_mask])) >= 2:
                pr_auc_e = float(average_precision_score(y_test[early_mask], prob_test[early_mask]))
            else:
                pr_auc_e = float("nan")
            f1_e = float(f1_score(y_test[early_mask], prob_test[early_mask] >= threshold, zero_division=0)) if len(y_test[early_mask]) > 0 else float("nan")
            f1_earlies.append(f1_e)
            pr_auc_earlies.append(pr_auc_e)

            if len(np.unique(y_test[late_mask])) >= 2:
                pr_auc_l = float(average_precision_score(y_test[late_mask], prob_test[late_mask]))
            else:
                pr_auc_l = float("nan")
            f1_l = float(f1_score(y_test[late_mask], prob_test[late_mask] >= threshold, zero_division=0)) if len(y_test[late_mask]) > 0 else float("nan")
            f1_lates.append(f1_l)
            pr_auc_lates.append(pr_auc_l)

            # Log to results.csv
            model_label = f"{MODEL_LABELS[group]}[{p_name}]({mode})"
            feat_set = get_feature_set_name(p_name)
            params_dict = {"detector": group, **rec["params"]}
            note = (
                f"params={_json(params_dict)};direction={direction};"
                f"pr_auc_35_42={pr_auc_e:.4f};pr_auc_43_49={pr_auc_l:.4f};threshold=tuned_on_val"
            )
            log_result(EXP, model=model_label, feature_set=feat_set, seed=seed, metrics=m, note=note)

            if seed == 0:
                seed0_test_probs[key] = prob_test
                seed0_models[key] = det

                # If this is the overall best prep for mode "best", dump seed 0 model
                if mode == "best" and p_name == overall_best_for_group[group]:
                    model_file = f"{TAG}_{group}.joblib"
                    ref_file = f"{TAG}_{group}_ref.npy"
                    np.save(MODEL_DIR / ref_file, np.sort(fit_scores))

                    # Drop large references if present
                    if hasattr(det, "_fit_X"):
                        det._fit_X = None
                    try:  # a model that cannot be saved must not kill the whole run
                        joblib.dump(det, MODEL_DIR / model_file)
                    except Exception as exc:
                        print(f"WARNING: could not save {model_file}: {type(exc).__name__}: {exc}", flush=True)

                    manifest_models[group] = {
                        "mode": mode,
                        "group": group,
                        "model_label": MODEL_LABELS[group],
                        "prep": p_name,
                        "params": params_dict,
                        "direction": int(direction),
                        "threshold": float(threshold),
                        "val_pr_auc": round(val_pr_auc, 4),
                        "test_pr_auc_seed0": round(m["pr_auc"], 4),
                        "test_f1_seed0": round(m["f1"], 4),
                        "model_file": model_file,
                        "ref_file": ref_file,
                    }

        summary_records.append({
            "mode": mode,
            "model": MODEL_LABELS[group],
            "prep": p_name,
            "params": _json(params_dict),
            "direction": direction,
            "threshold": round(float(threshold), 4),
            "val_pr_auc": round(float(val_pr_auc), 4),
            "precision_mean": round(float(np.mean(precisions)), 4),
            "precision_std": round(float(np.std(precisions)), 4),
            "recall_mean": round(float(np.mean(recalls)), 4),
            "recall_std": round(float(np.std(recalls)), 4),
            "f1_mean": round(float(np.mean(f1s)), 4),
            "f1_std": round(float(np.std(f1s)), 4),
            "pr_auc_mean": round(float(np.mean(pr_aucs)), 4),
            "pr_auc_std": round(float(np.std(pr_aucs)), 4),
            "pr_auc_35_42_mean": round(float(np.nanmean(pr_auc_earlies)), 4),
            "pr_auc_35_42_std": round(float(np.nanstd(pr_auc_earlies)), 4),
            "f1_35_42_mean": round(float(np.nanmean(f1_earlies)), 4),
            "f1_35_42_std": round(float(np.nanstd(f1_earlies)), 4),
            "pr_auc_43_49_mean": round(float(np.nanmean(pr_auc_lates)), 4),
            "pr_auc_43_49_std": round(float(np.nanstd(pr_auc_lates)), 4),
            "f1_43_49_mean": round(float(np.nanmean(f1_lates)), 4),
            "f1_43_49_std": round(float(np.nanstd(f1_lates)), 4),
        })

    # Save summary table (no duplicated column sets)
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(RESULTS_DIR / f"{TAG}_summary.csv", index=False, encoding="utf-8-sig")
    print(f"Saved results/{TAG}_summary.csv ({len(summary_df)} rows).", flush=True)

    # Save manifest.json
    manifest_data = {
        "experiment": EXP,
        "versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "seeds": [0, 1, 2],
        "models": manifest_models,
    }
    (MODEL_DIR / f"{TAG}_manifest.json").write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    print(f"Saved model/{TAG}_manifest.json ({len(manifest_models)} dumped models).", flush=True)

    # Save e6b_by_step.csv (for overall best prep of each group in mode "best", seed 0)
    step_rows = []
    for group in MODEL_GROUPS:
        best_p = overall_best_for_group[group]
        key = ("best", group, best_p)
        thresh = thresholds[key]
        prob = seed0_test_probs[key]
        df_step = evaluate_by_step(df_test, prob, threshold=thresh)
        df_step["model"] = MODEL_LABELS[group]
        df_step["prep"] = best_p
        df_step["mode"] = "best"
        step_rows.append(df_step)
    by_step_df = pd.concat(step_rows, ignore_index=True)
    by_step_df.to_csv(RESULTS_DIR / f"{TAG}_by_step.csv", index=False, encoding="utf-8-sig")
    print(f"Saved results/{TAG}_by_step.csv ({len(by_step_df)} rows).", flush=True)

    # -------------------------------------------------------------
    # Figures (English titles, matplotlib Agg, dpi 130)
    # -------------------------------------------------------------
    print("Generating figures...", flush=True)
    test_prevalence = float(np.mean(y_test == 1))

    # Figure 1: e6b_prep_compare.png (heatmap model x prep of best val PR-AUC)
    fig, ax = plt.subplots(figsize=(9, 6))
    heatmap_models = [MODEL_LABELS[g] for g in MODEL_GROUPS]
    heatmap_preps = prep.PREP_NAMES
    data_matrix = np.full((len(heatmap_models), len(heatmap_preps)), np.nan)

    for i, g in enumerate(MODEL_GROUPS):
        for j, p in enumerate(heatmap_preps):
            val = prep_compare_df.loc[prep_compare_df["model"] == MODEL_LABELS[g], p].values
            if len(val) > 0 and not np.isnan(val[0]):
                data_matrix[i, j] = val[0]

    im = ax.imshow(data_matrix, cmap="YlGnBu", aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Validation PR-AUC", rotation=270, labelpad=15)

    ax.set_xticks(np.arange(len(heatmap_preps)))
    ax.set_yticks(np.arange(len(heatmap_models)))
    ax.set_xticklabels(heatmap_preps, rotation=30, ha="right")
    ax.set_yticklabels(heatmap_models)

    # Annotate numeric values inside heatmap
    for i in range(len(heatmap_models)):
        for j in range(len(heatmap_preps)):
            v = data_matrix[i, j]
            if not np.isnan(v):
                text_color = "white" if v > np.nanmean(data_matrix) else "black"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", color=text_color, fontsize=8)
            else:
                ax.text(j, i, "N/A", ha="center", va="center", color="gray", fontsize=8)

    ax.set_title("E6b: Best Validation PR-AUC across Preprocessing Pipelines", fontsize=11, pad=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{TAG}_prep_compare.png", dpi=130)
    plt.close(fig)

    # Figure 2: e6b_test_pr_auc.png (best-mode test PR-AUC per model with E6 IF 0.446 ref)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    best_summary = summary_df[summary_df["mode"] == "best"].copy()
    model_best_rows = []
    for g in MODEL_GROUPS:
        best_p = overall_best_for_group[g]
        row_match = best_summary[(best_summary["model"] == MODEL_LABELS[g]) & (best_summary["prep"] == best_p)]
        if len(row_match) > 0:
            model_best_rows.append(row_match.iloc[0])

    m_df = pd.DataFrame(model_best_rows)
    xs = np.arange(len(m_df))
    bars = ax.bar(xs, m_df["pr_auc_mean"], yerr=m_df["pr_auc_std"], capsize=4, color="#4C72B0", alpha=0.9)

    for i, bar in enumerate(bars):
        prep_txt = m_df.iloc[i]["prep"]
        val_txt = f"{m_df.iloc[i]['pr_auc_mean']:.3f}\n[{prep_txt}]"
        ax.annotate(val_txt, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points", ha="center", fontsize=7.5)

    if EXP == "E6b":
        ax.axhline(0.446, color="tab:red", linestyle="--", label="E6 IForest Baseline (0.446)")
    ax.axhline(test_prevalence, color="gray", linestyle=":", label=f"Test Prevalence ({test_prevalence:.3f})")

    ax.set_xticks(xs)
    ax.set_xticklabels(m_df["model"], rotation=20, ha="right")
    ax.set(title=f"{EXP}: Test PR-AUC across Anomaly Detectors (Best Mode, Seeds 0-2)", ylabel="Test PR-AUC")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{TAG}_test_pr_auc.png", dpi=130)
    plt.close(fig)

    # Figure 3: e6b_by_step.png
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    steps = sorted(np.unique(df_test["time_step"]))
    counts = [int(np.sum((df_test["time_step"] == s) & (df_test["y"] == 1))) for s in steps]
    ax2.bar(steps, counts, color="lightgray", alpha=0.5, label="Illicit count")
    ax2.set_ylabel("Illicit count")

    for g in MODEL_GROUPS:
        sub = by_step_df[by_step_df["model"] == MODEL_LABELS[g]]
        if len(sub) > 0:
            ax1.plot(sub["time_step"], sub["pr_auc"], marker="o", label=MODEL_LABELS[g])

    ax1.axvline(DARK_MARKET_STEP, color="black", linestyle="--", label=f"Dark Market (Step {DARK_MARKET_STEP})")
    ax1.set(title=f"{EXP}: Test PR-AUC by Time Step (Seed 0, Best Mode)", xlabel="Time Step", ylabel="PR-AUC")
    ax1.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{TAG}_by_step.png", dpi=130)
    plt.close(fig)

    # Figure 4: e6b_gmm_components.png (per component size vs logdet volume, illicit rate color)
    gmm_best_p = overall_best_for_group["gmm_component"]
    gmm_key = ("best", "gmm_component", gmm_best_p)
    gmm_det = seed0_models[gmm_key]

    X_fu = prep_data_full[(gmm_best_p, 0)]
    Z_fu = gmm_det.pca_model.transform(X_fu) if gmm_det.pca_model is not None else X_fu
    c_pred_full = gmm_det.gmm.predict(Z_fu)
    y_full = df_full["y"].values

    comp_sizes = []
    comp_vols = []
    comp_rates = []
    n_components = len(gmm_det.component_logdets_)

    for c in range(n_components):
        mask_c = (c_pred_full == c)
        c_size = int(np.sum(mask_c))
        c_vol = float(gmm_det.component_logdets_[c])
        labeled_mask = mask_c & pd.notna(y_full)
        if np.sum(labeled_mask) > 0:
            c_rate = float(np.mean(y_full[labeled_mask] == 1))
        else:
            c_rate = 0.0
        comp_sizes.append(c_size)
        comp_vols.append(c_vol)
        comp_rates.append(c_rate)

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(comp_sizes, comp_vols, c=comp_rates, cmap="viridis", s=180, edgecolors="black", linewidths=1.2)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Post-hoc Illicit Rate (Steps 1-34)", rotation=270, labelpad=15)

    for c in range(n_components):
        ax.annotate(f"C{c}", (comp_sizes[c], comp_vols[c]), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)

    ax.set(
        title=f"E6b: GMM Components Size vs Log-Det Volume ({gmm_best_p}, k={n_components})",
        xlabel="Component Size (Transactions in Steps 1-34)",
        ylabel="Log-Determinant Volume (0.5 * log|det(Sigma_c)|)",
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{TAG}_gmm_components.png", dpi=130)
    plt.close(fig)

    print("E6b improved experiment completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
