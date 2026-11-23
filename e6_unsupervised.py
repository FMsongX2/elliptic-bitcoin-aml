from common import BASE
from common import (
    load_dataset,
    feature_cols,
    evaluate,
    evaluate_by_step,
    tune_threshold,
    log_result,
    clear_results,
    TRAIN_STEPS,
    VAL_STEPS,
    TEST_STEPS,
    DARK_MARKET_STEP,
)

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, silhouette_score

import detectors

# Directories configurable for tests
FIG_DIR = BASE / "figures"
MODEL_DIR = BASE / "models"
RESULTS_DIR = BASE / "results"

GRID_FIT_MAX_STEP = 30

GROUPS = {
    "kmeans": ["kmeans"],
    "dbscan": ["dbscan"],
    "gmm": ["gmm", "bgmm"],
    "iforest": ["iforest"],
    "lof": ["lof"],
}

GROUP_NAMES = {
    "kmeans": "KMeans",
    "dbscan": "DBSCAN",
    "gmm": "GMM",
    "iforest": "IForest",
    "lof": "LOF",
}


def _json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def to_prob(fit_scores, scores):
    """Convert anomaly scores to probabilities via empirical CDF on fit_scores."""
    fit_sorted = np.sort(np.asarray(fit_scores, dtype=float))
    scores = np.asarray(scores, dtype=float)
    n_fit = len(fit_sorted)
    if n_fit == 0:
        return np.zeros_like(scores, dtype=float)
    return np.searchsorted(fit_sorted, scores, side="right") / float(n_fit)


def _grid_specs(fast=False):
    specs = []
    ks = [2, 4] if fast else [2, 4, 8, 16, 32, 64]
    for alg, k in ([('KMeans', 2), ('MiniBatchKMeans', 4)] if fast else
                   [(a, k) for a in ('KMeans', 'MiniBatchKMeans') for k in ks]):
        specs.append(("kmeans", {"algorithm": alg, "n_clusters": k}))
    for pca, mins, q in ([(10, 5, .95), (10, 5, .99)] if fast else
                         [(d, m, q) for d in (10, 20) for m in (5, 10, 20)
                          for q in (.90, .95, .98, .99)]):
        specs.append(("dbscan", {"pca": pca, "min_samples": mins,
                                  "eps_quantile": q}))
    for n, cov, pca in ([(2, 'full', 10), (4, 'diag', 20)] if fast else
                        [(n, c, d) for d in (10, 20) for n in (1, 2, 4, 8, 12, 16)
                         for c in ('full', 'tied', 'diag', 'spherical')]):
        specs.append(("gmm", {"pca": pca, "n_components": n, "covariance_type": cov}))
    for prior in ([.01] if fast else [.01, 1, 100]):
        specs.append(("bgmm", {"pca": 10 if fast else 20, "n_components": 16,
                               "weight_concentration_prior": prior, "max_iter": 100}))
    for trees, samples, features in ([(100, 256, .5), (100, 1024, 1.)] if fast else
                                     [(e, s, f) for e in (100, 300)
                                      for s in (256, 1024, 4096) for f in (.5, 1.)]):
        specs.append(("iforest", {"n_estimators": trees, "max_samples": samples,
                                   "max_features": features}))
    for pca, neighbors in ([(10, 10), (10, 20)] if fast else
                           [(d, k) for d in (10, 20) for k in (10, 20, 50, 100)]):
        specs.append(("lof", {"pca": pca, "n_neighbors": neighbors}))
    return specs


def _intrinsic(detector, name):
    if name == "kmeans":
        Z = detector._transform(detector._fit_X)
        labels = detector.model.labels_
        if len(set(labels)) > 1 and len(set(labels)) < len(Z):
            sil = float(silhouette_score(Z, labels, sample_size=min(10000, len(Z)), random_state=0))
        else:
            sil = float("nan")
        return "inertia_silhouette", [float(detector.model.inertia_), sil]
    if name == "dbscan":
        labels = detector.model.labels_
        return "clusters_noise_fraction", [int(len(set(labels) - {-1})), float(np.mean(labels == -1))]
    if name in ("gmm", "bgmm"):
        Z = detector._transform(detector._fit_X)
        if name == "gmm":
            return "bic_aic", [float(detector.model.bic(Z)), float(detector.model.aic(Z))]
        return "effective_components", int(np.sum(detector.model.weights_ > .01))
    return "none", None


def _append_row(path, fieldnames, row):
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new:
            w.writeheader()
        w.writerow(row)


def main(argv=None):
    global GRID_FIT_MAX_STEP
    parser = argparse.ArgumentParser(description="E6 Unsupervised experiment on team conventions.")
    parser.add_argument("--fast", action="store_true", help="Run fast smoke test (subsample splits and configs)")
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Elliptic dataset...", flush=True)
    df = load_dataset(labeled_only=False)
    features = feature_cols(df, "all")
    print(f"Loaded {len(df)} transactions, {len(features)} features.", flush=True)

    X_all = df[features].values.astype(np.float32)
    y_all = df["y"].values
    t_all = df["time_step"].values

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
    print(f"Splits: fit_train={train_mask.sum()} (max t={GRID_FIT_MAX_STEP}), val={val_mask.sum()}, test={test_mask.sum()}, fit_full={full_mask.sum()}", flush=True)

    X_train = X_all[train_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask].astype(int)
    X_test, y_test = X_all[test_mask], y_all[test_mask].astype(int)
    X_full = X_all[full_mask]
    df_test = df[test_mask].copy()

    # -------------------------------------------------------------
    # Grid stage (resumable)
    # -------------------------------------------------------------
    grid_csv = RESULTS_DIR / "e6_grid.csv"
    grid_fields = [
        "config_id", "model", "params", "val_pr_auc", "val_pr_auc_neg",
        "intrinsic_metric", "intrinsic_value", "extra_json",
        "fit_seconds", "fit_max_t", "val_scores_json"
    ]
    existing = {}
    if grid_csv.exists() and grid_csv.stat().st_size > 0:
        with grid_csv.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                existing[r["config_id"]] = r

    specs = _grid_specs(fast=args.fast)
    records = []
    print(f"Running grid search ({len(specs)} configs, {len(existing)} cached)...", flush=True)

    for index, (name, params) in enumerate(specs, 1):
        config_id = name + ":" + _json(params)
        if config_id in existing:
            r = existing[config_id]
            val_scores = np.asarray(json.loads(r["val_scores_json"]), dtype=float)
            pr_auc = float(r["val_pr_auc"])
            pr_auc_neg = float(r.get("val_pr_auc_neg", average_precision_score(y_val, -val_scores)))
            ivalue = json.loads(r["intrinsic_value"]) if r["intrinsic_value"] != "null" else None
            extras = json.loads(r.get("extra_json", "{}"))
            records.append({
                "config_id": config_id,
                "model": name,
                "params": params,
                "val_scores": val_scores,
                "val_pr_auc": pr_auc,
                "val_pr_auc_neg": pr_auc_neg,
                "intrinsic_metric": r["intrinsic_metric"],
                "intrinsic_value": ivalue,
                "fit_seconds": float(r["fit_seconds"]),
                "detector": None,
                "extra": extras,
            })
            print(f"[{index}/{len(specs)}] {name} {params} val_pr_auc={pr_auc:.4f} (cached)", flush=True)
            continue

        t0 = time.perf_counter()
        detector = detectors.make_detector(name, params, seed=0).fit(X_train)
        fit_seconds = time.perf_counter() - t0
        val_scores = detector.score(X_val)
        pr_auc = float(average_precision_score(y_val, val_scores))
        pr_auc_neg = float(average_precision_score(y_val, -val_scores))
        imetric, ivalue = _intrinsic(detector, name)

        extras = {}
        if name == "dbscan":
            extras["eps"] = getattr(detector, "eps_", None)
        if name == "bgmm":
            extras["sorted_weights"] = np.sort(detector.model.weights_)[::-1].tolist()

        row = {
            "config_id": config_id,
            "model": name,
            "params": _json(params),
            "val_pr_auc": pr_auc,
            "val_pr_auc_neg": pr_auc_neg,
            "intrinsic_metric": imetric,
            "intrinsic_value": _json(ivalue) if ivalue is not None else "null",
            "extra_json": _json(extras),
            "fit_seconds": round(fit_seconds, 3),
            "fit_max_t": GRID_FIT_MAX_STEP,
            "val_scores_json": _json(val_scores.tolist()),
        }
        _append_row(grid_csv, grid_fields, row)
        records.append({
            "config_id": config_id,
            "model": name,
            "params": params,
            "val_scores": val_scores,
            "val_pr_auc": pr_auc,
            "val_pr_auc_neg": pr_auc_neg,
            "intrinsic_metric": imetric,
            "intrinsic_value": ivalue,
            "fit_seconds": fit_seconds,
            "detector": detector,
            "extra": extras,
        })
        print(f"[{index}/{len(specs)}] {name} {params} val_pr_auc={pr_auc:.4f} ({fit_seconds:.2f}s)", flush=True)

    # -------------------------------------------------------------
    # Selection stage per group
    # -------------------------------------------------------------
    selected = {"best": {}, "anomaly": {}}
    intrinsic_selected = {}

    for group, members in GROUPS.items():
        candidates = [r for r in records if r["model"] in members]
        if not candidates:
            continue

        # Anomaly mode: direction +1 only, max val_pr_auc
        best_anomaly = max(candidates, key=lambda r: r["val_pr_auc"])
        selected["anomaly"][group] = {"record": best_anomaly, "direction": 1}

        # Best mode: direction +1 or -1, max val PR-AUC
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
        selected["best"][group] = {"record": best_rec, "direction": best_dir}

        # Intrinsic criteria
        int_cands = [r for r in candidates if r["intrinsic_value"] is not None]
        if int_cands:
            if group == "kmeans":
                intrinsic_selected[group] = {
                    "criterion": "max silhouette",
                    "record": max(int_cands, key=lambda r: (r["intrinsic_value"][1] if not np.isnan(r["intrinsic_value"][1]) else -999)),
                }
            elif group == "gmm":
                gmm_only = [r for r in int_cands if r["model"] == "gmm"]
                if gmm_only:
                    intrinsic_selected[group] = {
                        "criterion": "min BIC",
                        "record": min(gmm_only, key=lambda r: r["intrinsic_value"][0]),
                    }
            elif group == "dbscan":
                intrinsic_selected[group] = {
                    "criterion": "noise closest 5%",
                    "record": min(int_cands, key=lambda r: abs(r["intrinsic_value"][1] - 0.05)),
                }

    # Save e6_intrinsic.csv
    intrinsic_rows = []
    for group, info in intrinsic_selected.items():
        int_rec = info["record"]
        val_rec = selected["best"][group]["record"]
        val_dir = selected["best"][group]["direction"]
        val_auc = val_rec["val_pr_auc"] if val_dir == 1 else val_rec["val_pr_auc_neg"]
        int_metric_val = int_rec["intrinsic_value"][1] if group in ("kmeans", "dbscan") else int_rec["intrinsic_value"][0]
        intrinsic_rows.append({
            "group": group,
            "intrinsic_criterion": info["criterion"],
            "intrinsic_model": int_rec["model"],
            "intrinsic_params": _json(int_rec["params"]),
            "intrinsic_metric_value": int_metric_val,
            "intrinsic_val_pr_auc": int_rec["val_pr_auc"],
            "val_selected_model": val_rec["model"],
            "val_selected_params": _json(val_rec["params"]),
            "val_selected_direction": val_dir,
            "val_selected_pr_auc": val_auc,
            "diff_val_pr_auc": int_rec["val_pr_auc"] - val_auc,
        })
    pd.DataFrame(intrinsic_rows).to_csv(RESULTS_DIR / "e6_intrinsic.csv", index=False, encoding="utf-8-sig")

    # Tune thresholds on val using fit_train model
    thresholds = {}
    for mode in ("anomaly", "best"):
        for group in GROUPS:
            choice = selected[mode][group]
            rec = choice["record"]
            direction = choice["direction"]
            det = rec["detector"]
            if det is None:
                det = detectors.make_detector(rec["model"], rec["params"], seed=0).fit(X_train)
            fit_scores = direction * det.score(det._fit_X)
            val_scores = direction * rec["val_scores"]
            prob_val = to_prob(fit_scores, val_scores)
            thresholds[f"{mode}:{group}"] = tune_threshold(y_val, prob_val)

    # -------------------------------------------------------------
    # Final Stage: refit on fit_full with seeds [0, 1, 2]
    # -------------------------------------------------------------
    print("Starting final stage: refitting selected models on fit_full...", flush=True)
    clear_results("E6")

    early_mask = (df_test["time_step"] <= 42).values
    late_mask = (df_test["time_step"] >= 43).values

    summary_records = []
    seed0_test_probs = {}
    seed0_models = {}
    manifest_models = {}

    for mode in ("best", "anomaly"):
        for group in GROUPS:
            choice = selected[mode][group]
            rec = choice["record"]
            direction = choice["direction"]
            threshold = thresholds[f"{mode}:{group}"]
            val_pr_auc = rec["val_pr_auc"] if direction == 1 else rec["val_pr_auc_neg"]

            precisions, recalls, f1s, pr_aucs = [], [], [], []
            f1_earlies, pr_auc_earlies = [], []
            f1_lates, pr_auc_lates = [], []

            for seed in [0, 1, 2]:
                det = detectors.make_detector(rec["model"], rec["params"], seed=seed).fit(X_full)
                fit_scores = direction * det.score(det._fit_X)
                test_scores = direction * det.score(X_test)
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

                params_with_det = {"detector": rec["model"], **rec["params"]}
                note = f"params={_json(params_with_det)};direction={direction};pr_auc_35_42={pr_auc_e:.4f};pr_auc_43_49={pr_auc_l:.4f};threshold=tuned_on_val"
                model_label = f"{GROUP_NAMES[group]}({mode})"
                log_result("E6", model=model_label, feature_set="all", seed=seed, metrics=m, note=note)

                if seed == 0:
                    seed0_test_probs[f"{mode}:{group}"] = prob_test
                    seed0_models[f"{mode}:{group}"] = det

                    # Save ref scores and joblib model
                    ref_file = f"{mode}_{group}_ref.npy"
                    model_file = f"{mode}_{group}.joblib"
                    np.save(MODEL_DIR / ref_file, np.sort(fit_scores))

                    # Drop _fit_X before saving detector
                    det._fit_X = None
                    joblib.dump(det, MODEL_DIR / model_file)

                    manifest_models[f"{mode}_{group}"] = {
                        "mode": mode,
                        "group": group,
                        "model_name": GROUP_NAMES[group],
                        "detector": rec["model"],
                        "params": params_with_det,
                        "direction": int(direction),
                        "threshold": float(threshold),
                        "ref_file": ref_file,
                        "model_file": model_file,
                        "val_pr_auc": round(val_pr_auc, 4),
                        "test_pr_auc_seed0": round(m["pr_auc"], 4),
                    }

            params_with_det = {"detector": rec["model"], **rec["params"]}
            summary_records.append({
                "mode": mode,
                "model": group,
                "params": _json(params_with_det),
                "direction": direction,
                "val_pr_auc": round(val_pr_auc, 4),
                "precision_mean": round(float(np.mean(precisions)), 4),
                "precision_std": round(float(np.std(precisions)), 4),
                "recall_mean": round(float(np.mean(recalls)), 4),
                "recall_std": round(float(np.std(recalls)), 4),
                "f1_mean": round(float(np.mean(f1s)), 4),
                "f1_std": round(float(np.std(f1s)), 4),
                "pr_auc_mean": round(float(np.mean(pr_aucs)), 4),
                "pr_auc_std": round(float(np.std(pr_aucs)), 4),
                "f1_35_42_mean": round(float(np.nanmean(f1_earlies)), 4),
                "f1_35_42_std": round(float(np.nanstd(f1_earlies)), 4),
                "pr_auc_35_42_mean": round(float(np.nanmean(pr_auc_earlies)), 4),
                "pr_auc_35_42_std": round(float(np.nanstd(pr_auc_earlies)), 4),
                "f1_43_49_mean": round(float(np.nanmean(f1_lates)), 4),
                "f1_43_49_std": round(float(np.nanstd(f1_lates)), 4),
                "pr_auc_43_49_mean": round(float(np.nanmean(pr_auc_lates)), 4),
                "pr_auc_43_49_std": round(float(np.nanstd(pr_auc_lates)), 4),
                # Also include test_ prefixed keys
                "test_precision_mean": round(float(np.mean(precisions)), 4),
                "test_precision_std": round(float(np.std(precisions)), 4),
                "test_recall_mean": round(float(np.mean(recalls)), 4),
                "test_recall_std": round(float(np.std(recalls)), 4),
                "test_f1_mean": round(float(np.mean(f1s)), 4),
                "test_f1_std": round(float(np.std(f1s)), 4),
                "test_pr_auc_mean": round(float(np.mean(pr_aucs)), 4),
                "test_pr_auc_std": round(float(np.std(pr_aucs)), 4),
                "test_f1_35_42_mean": round(float(np.nanmean(f1_earlies)), 4),
                "test_f1_35_42_std": round(float(np.nanstd(f1_earlies)), 4),
                "test_pr_auc_35_42_mean": round(float(np.nanmean(pr_auc_earlies)), 4),
                "test_pr_auc_35_42_std": round(float(np.nanstd(pr_auc_earlies)), 4),
                "test_f1_43_49_mean": round(float(np.nanmean(f1_lates)), 4),
                "test_f1_43_49_std": round(float(np.nanstd(f1_lates)), 4),
                "test_pr_auc_43_49_mean": round(float(np.nanmean(pr_auc_lates)), 4),
                "test_pr_auc_43_49_std": round(float(np.nanstd(pr_auc_lates)), 4),
            })

    # Save manifest.json
    manifest_data = {
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
    (MODEL_DIR / "manifest.json").write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

    # Save e6_summary.csv
    pd.DataFrame(summary_records).to_csv(RESULTS_DIR / "e6_summary.csv", index=False, encoding="utf-8-sig")

    # Save e6_by_step.csv (for mode best, seed 0)
    step_rows = []
    for group in GROUPS:
        thresh = thresholds[f"best:{group}"]
        prob = seed0_test_probs[f"best:{group}"]
        df_step = evaluate_by_step(df_test, prob, threshold=thresh)
        df_step["model"] = group
        df_step["mode"] = "best"
        step_rows.append(df_step)
    by_step_df = pd.concat(step_rows, ignore_index=True)
    by_step_df.to_csv(RESULTS_DIR / "e6_by_step.csv", index=False, encoding="utf-8-sig")

    # -------------------------------------------------------------
    # Figures (English titles, matplotlib Agg, dpi 130)
    # -------------------------------------------------------------
    print("Generating figures...", flush=True)
    test_prevalence = float(np.mean(y_test == 1))

    # Figure 1: e6_direction.png
    fig, ax = plt.subplots(figsize=(8, 4.8))
    model_keys = list(GROUPS.keys())
    x = np.arange(len(model_keys))
    width = 0.35

    summary_df = pd.DataFrame(summary_records)
    anom_vals = [summary_df[(summary_df["mode"] == "anomaly") & (summary_df["model"] == g)]["pr_auc_mean"].values[0] for g in model_keys]
    best_vals = [summary_df[(summary_df["mode"] == "best") & (summary_df["model"] == g)]["pr_auc_mean"].values[0] for g in model_keys]
    best_dirs = [summary_df[(summary_df["mode"] == "best") & (summary_df["model"] == g)]["direction"].values[0] for g in model_keys]

    b1 = ax.bar(x - width / 2, anom_vals, width, label="Anomaly (+1)", color="#4C72B0")
    b2 = ax.bar(x + width / 2, best_vals, width, label="Best (+1 / -1)", color="#55A868")

    for bar, d in zip(b2, best_dirs):
        ax.annotate(f"{int(d):+d}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    ax.axhline(test_prevalence, color="gray", linestyle="--", label=f"Test prevalence ({test_prevalence:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_NAMES[g] for g in model_keys])
    ax.set(title="Test PR-AUC: Anomaly Direction (+1) vs Best Direction", ylabel="Test PR-AUC")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e6_direction.png", dpi=130)
    plt.close(fig)

    # Figure 2: e6_pr_curves.png
    fig, ax = plt.subplots(figsize=(7, 5))
    for group in model_keys:
        prob = seed0_test_probs[f"best:{group}"]
        auc = average_precision_score(y_test, prob) if len(np.unique(y_test)) >= 2 else 0.0
        p, r, _ = precision_recall_curve(y_test, prob)
        ax.plot(r, p, label=f"{GROUP_NAMES[group]} (PR-AUC={auc:.3f})")

    ax.axhline(test_prevalence, color="gray", linestyle="--", label=f"Test prevalence ({test_prevalence:.3f})")
    ax.set(title="Best Mode Test Precision-Recall Curves (Seed 0)", xlabel="Recall", ylabel="Precision")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e6_pr_curves.png", dpi=130)
    plt.close(fig)

    # Figure 3: e6_by_step.png
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    steps = sorted(np.unique(df_test["time_step"]))
    counts = [int(np.sum((df_test["time_step"] == s) & (df_test["y"] == 1))) for s in steps]
    ax2.bar(steps, counts, color="lightgray", alpha=0.5, label="Illicit count")
    ax2.set_ylabel("Illicit count")

    for group in model_keys:
        sub = by_step_df[by_step_df["model"] == group]
        ax1.plot(sub["time_step"], sub["pr_auc"], marker="o", label=GROUP_NAMES[group])

    ax1.axvline(DARK_MARKET_STEP, color="black", linestyle="--", label=f"Dark Market (Step {DARK_MARKET_STEP})")
    ax1.set(title="Best Mode Test PR-AUC by Time Step (Seed 0)", xlabel="Time Step", ylabel="PR-AUC")
    ax1.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e6_by_step.png", dpi=130)
    plt.close(fig)

    # Figure 4: e6_kmeans_clusters.png
    km_det = seed0_models["best:kmeans"]
    km_clusters = km_det.model.predict(km_det._transform(X_test))
    cluster_rates = []
    for c in np.unique(km_clusters):
        c_mask = (km_clusters == c)
        c_n = int(np.sum(c_mask))
        c_rate = float(np.mean(y_test[c_mask])) if c_n > 0 else 0.0
        cluster_rates.append({"cluster": int(c), "n": c_n, "rate": c_rate})

    cluster_rates.sort(key=lambda x: x["rate"])
    fig, ax = plt.subplots(figsize=(8, 4.8))
    xs = np.arange(len(cluster_rates))
    bars = ax.bar(xs, [c["rate"] for c in cluster_rates], color="#4C72B0")
    for x_i, c in zip(xs, cluster_rates):
        ax.annotate(f"n={c['n']}", (x_i, c["rate"]), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)

    ax.axhline(test_prevalence, color="tab:red", linestyle="--", label=f"Test prevalence ({test_prevalence:.3f})")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"C{c['cluster']}" for c in cluster_rates])
    ax.set(title="K-Means Clusters Sorted by Test Illicit Rate", xlabel="Cluster (sorted by illicit rate)", ylabel="Illicit Rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e6_kmeans_clusters.png", dpi=130)
    plt.close(fig)

    print("E6 unsupervised experiment completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
