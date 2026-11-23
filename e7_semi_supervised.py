"""E7 semi-supervised learning / label efficiency (setting B: 57 transaction-time features).

Question: with only a fraction of the labels, how close do we get to fully supervised LightGBM,
and does using unlabeled transactions (unknown + not-yet-labeled) help?

Protocol (team common.py, same as E6/E6b):
- Label pool = labeled transactions of the fit steps. A fraction `frac` of the pool is "queried"
  (label revealed). The rest of the pool and all unknown rows are used as unlabeled data.
- Fit on steps 1-30 -> threshold tuned on labeled val 31-34 (tune_threshold).
- Refit on steps 1-34 (same frac, same seed) -> evaluate on labeled test 35-49 with that threshold.
- Seeds 0, 1, 2 (label selection and model seed).

Methods:
  sup_random       LightGBM (team E2 params) on a random frac of the pool. No unlabeled data.
  sup_random_notp  same without the time-proxy features agg_6/7/9 (predecessor block height).
  self_training    sklearn SelfTrainingClassifier(LightGBM), threshold 0.95, 5 rounds.
  label_spreading  LabelSpreading(knn) on quantile-normal + PCA-20 features.
  cluster_rep      Lec9: K-Means on the pool with k = label budget; label only the pool row
                   closest to each centroid; LightGBM on those k rows.
  cluster_prop     Lec9: propagate each representative's label to the 20% of rows (pool and
                   unknown) of its cluster closest to the centroid; LightGBM on the propagated set.
  active_unc       Lorenz et al. (2020): 10% of the budget at random, then 9 rounds of
                   uncertainty sampling (LightGBM prob closest to 0.5) until the budget is used.
cluster_* and active_unc run only for frac <= 0.10 (with k = budget they are pointless at 50-100%).

Outputs: results/e7_runs.csv (resumable), results/e7_summary.csv, results.csv rows "E7",
figures/e7_*.png, models/e7_<method>_<frac>.joblib (seed 0, fit 1-34).
"""
import argparse
import csv
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, f1_score
from sklearn.preprocessing import QuantileTransformer
from sklearn.semi_supervised import LabelSpreading, SelfTrainingClassifier

from common import BASE
import common
from common import (DARK_MARKET_STEP, TEST_STEPS, TRAIN_STEPS, VAL_STEPS, clear_results, evaluate,
                    feature_cols, load_dataset, log_result, tune_threshold)

FIG_DIR = BASE / "figures"
MODEL_DIR = BASE / "models"
RESULTS_DIR = BASE / "results"

TIME_PROXY = ["agg_6", "agg_7", "agg_9"]
FRACS = [0.01, 0.05, 0.10, 0.50, 1.00]
SEEDS = [0, 1, 2]
METHODS = ["sup_random", "sup_random_notp", "self_training", "label_spreading",
           "cluster_rep", "cluster_prop", "active_unc"]
SMALL_ONLY = {"cluster_rep", "cluster_prop", "active_unc"}
MIN_LABELS = 20
UNLABELED_CAP = 60000  # unlabeled rows used by self_training / label_spreading (memory, time)

LABELS = {
    "sup_random": "LightGBM, random labels",
    "sup_random_notp": "LightGBM, random labels, no agg_6/7/9",
    "self_training": "Self-training (LightGBM)",
    "label_spreading": "Label Spreading (kNN)",
    "cluster_rep": "Cluster representatives (Lec9)",
    "cluster_prop": "Cluster propagation 20% (Lec9)",
    "active_unc": "Active learning, uncertainty",
}
RUN_FIELDS = ["method", "frac", "seed", "n_labels", "n_illicit_labels", "val_pr_auc", "threshold",
              "precision", "recall", "f1", "pr_auc", "pr_auc_35_42", "pr_auc_43_49",
              "f1_35_42", "f1_43_49", "fit_seconds"]


# ---------------------------------------------------------------- models
class PriorModel:
    """Fallback when the queried labels contain one class only: constant probability."""

    def __init__(self, p):
        self.p = float(p)

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1 - self.p), np.full(n, self.p)])


def make_lgbm(seed, n_estimators=500):
    return lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=0.05, num_leaves=31,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                              random_state=seed, n_jobs=-1, verbose=-1)


def fit_lgbm(X, y, seed, n_estimators=500):
    if len(np.unique(y)) < 2:
        return PriorModel(np.mean(y) if len(y) else 0.0)
    return make_lgbm(seed, n_estimators).fit(X, y)


class SpreadModel:
    """Quantile + PCA embedding with a fitted LabelSpreading model."""

    def __init__(self, emb, ls, prior):
        self.emb, self.ls, self.prior = emb, ls, float(prior)

    def predict_proba(self, X):
        """Rows whose kNN neighbours received no label (0/0 -> NaN) get the labeled illicit share."""
        with np.errstate(invalid="ignore", divide="ignore"):
            p = self.ls.predict_proba(self.emb.transform(X))
        bad = ~np.isfinite(p).all(axis=1)
        p[bad] = [1 - self.prior, self.prior]
        return p


class Embedding:
    """QuantileTransformer(normal) + PCA, fit on all rows of the fit steps (no labels)."""

    def __init__(self, seed, n_pca=20):
        self.seed, self.n_pca = seed, n_pca

    def fit(self, X):
        self.qt = QuantileTransformer(output_distribution="normal", n_quantiles=min(1000, len(X)),
                                      subsample=100000, random_state=self.seed).fit(X)
        Z = self.qt.transform(X)
        self.pca = PCA(n_components=min(self.n_pca, Z.shape[1], len(Z)), random_state=self.seed).fit(Z)
        return self

    def transform(self, X):
        return self.pca.transform(self.qt.transform(X)).astype(np.float32)


# ---------------------------------------------------------------- one training run
def train_method(method, X_all, pool_idx, pool_y, unk_idx, budget, seed, emb_cache):
    """Return (model, n_labels, n_illicit_labels). Row indices refer to X_all (fit steps only)."""
    rng = np.random.RandomState(seed)

    def random_query(n):
        return rng.choice(len(pool_idx), size=min(n, len(pool_idx)), replace=False)

    def unlabeled_rows(queried_pos):
        rest = np.setdiff1d(np.arange(len(pool_idx)), queried_pos)
        unl = np.concatenate([pool_idx[rest], unk_idx])
        if len(unl) > UNLABELED_CAP:
            unl = rng.choice(unl, UNLABELED_CAP, replace=False)
        return unl

    if method in ("sup_random", "sup_random_notp"):
        q = random_query(budget)
        return fit_lgbm(X_all[pool_idx[q]], pool_y[q], seed), len(q), int(pool_y[q].sum())

    if method == "self_training":
        q = random_query(budget)
        if len(np.unique(pool_y[q])) < 2:
            return PriorModel(pool_y[q].mean()), len(q), int(pool_y[q].sum())
        unl = unlabeled_rows(q)
        X = np.vstack([X_all[pool_idx[q]], X_all[unl]])
        y = np.concatenate([pool_y[q], -np.ones(len(unl), dtype=int)])
        st = SelfTrainingClassifier(make_lgbm(seed), threshold=0.95, max_iter=5).fit(X, y)
        return st, len(q), int(pool_y[q].sum())

    # embedding (quantile + PCA) on all fit rows, shared across methods for the same fit
    if "emb" not in emb_cache:
        emb = Embedding(seed=0).fit(X_all)
        emb_cache["emb"] = emb
        emb_cache["Z"] = emb.transform(X_all)
    emb, Z = emb_cache["emb"], emb_cache["Z"]

    if method == "label_spreading":
        q = random_query(budget)
        if len(np.unique(pool_y[q])) < 2:
            return PriorModel(pool_y[q].mean()), len(q), int(pool_y[q].sum())
        unl = unlabeled_rows(q)
        Zt = np.vstack([Z[pool_idx[q]], Z[unl]])
        y = np.concatenate([pool_y[q], -np.ones(len(unl), dtype=int)])
        ls = LabelSpreading(kernel="knn", n_neighbors=10, alpha=0.2, max_iter=50).fit(Zt, y)
        return SpreadModel(emb, ls, pool_y[q].mean()), len(q), int(pool_y[q].sum())

    if method in ("cluster_rep", "cluster_prop"):
        k = min(budget, len(pool_idx))
        Zp = Z[pool_idx]
        km_cls = KMeans if k <= 500 else MiniBatchKMeans
        kw = dict(n_init=3) if km_cls is KMeans else dict(n_init=3, batch_size=4096)
        km = km_cls(n_clusters=k, random_state=seed, **kw).fit(Zp)
        d_pool = np.linalg.norm(Zp - km.cluster_centers_[km.labels_], axis=1)
        reps = np.full(k, -1)
        order = np.lexsort((d_pool, km.labels_))  # by cluster, then distance
        first = np.r_[True, km.labels_[order][1:] != km.labels_[order][:-1]]
        reps[km.labels_[order][first]] = order[first]
        reps = reps[reps >= 0]
        rep_y = pool_y[reps]
        if method == "cluster_rep":
            return fit_lgbm(X_all[pool_idx[reps]], rep_y, seed), len(reps), int(rep_y.sum())
        # propagation: rows (pool + unknown) in the closest 20% of their cluster get the rep's label
        cl_label = np.full(k, -1)
        cl_label[km.labels_[reps]] = rep_y
        rows = np.concatenate([pool_idx, unk_idx])
        lab = km.predict(Z[rows])
        dist = np.linalg.norm(Z[rows] - km.cluster_centers_[lab], axis=1)
        cut = pd.Series(dist).groupby(lab).transform(lambda s: s.quantile(0.2)).to_numpy()
        keep = (dist <= cut) & (cl_label[lab] >= 0)
        Xp = np.vstack([X_all[rows[keep]], X_all[pool_idx[reps]]])
        yp = np.concatenate([cl_label[lab[keep]], rep_y])
        return fit_lgbm(Xp, yp, seed), len(reps), int(rep_y.sum())

    if method == "active_unc":
        n0 = max(MIN_LABELS // 2, budget // 10)
        q = list(random_query(n0))
        step = max(1, (budget - len(q)) // 9)
        while len(q) < budget:
            m = fit_lgbm(X_all[pool_idx[q]], pool_y[q], seed, n_estimators=200)
            rest = np.setdiff1d(np.arange(len(pool_idx)), q)
            n_new = min(step, budget - len(q), len(rest))
            if n_new <= 0:
                break
            if isinstance(m, PriorModel):
                new = rng.choice(rest, n_new, replace=False)
            else:
                p = m.predict_proba(X_all[pool_idx[rest]])[:, 1]
                new = rest[np.argsort(np.abs(p - 0.5))[:n_new]]
            q.extend(new.tolist())
        q = np.asarray(q)
        return fit_lgbm(X_all[pool_idx[q]], pool_y[q], seed), len(q), int(pool_y[q].sum())

    raise ValueError(method)


# ---------------------------------------------------------------- driver
def _fit_split(df, steps, cols):
    m = df["time_step"].isin(list(steps)).to_numpy()
    sub = df[m]
    X = sub[cols].to_numpy(dtype=np.float32)
    y = sub["y"].to_numpy()
    pool = np.flatnonzero(pd.notna(y))
    unk = np.flatnonzero(pd.isna(y))
    return X, pool, y[pool].astype(int), unk


def _period_metrics(y, p, t, thr):
    out = {}
    for name, mask in [("35_42", t < DARK_MARKET_STEP), ("43_49", t >= DARK_MARKET_STEP)]:
        yy, pp = y[mask], p[mask]
        out[f"pr_auc_{name}"] = float(average_precision_score(yy, pp)) if len(np.unique(yy)) == 2 else float("nan")
        out[f"f1_{name}"] = float(f1_score(yy, pp >= thr, zero_division=0)) if len(yy) else float("nan")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="E7 semi-supervised / label efficiency, setting B")
    ap.add_argument("--fast", action="store_true", help="smoke test: subsample rows, 2 fracs, 1 seed")
    ap.add_argument("--methods", nargs="*", default=METHODS)
    ap.add_argument("--fracs", nargs="*", type=float, default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    args = ap.parse_args(argv)
    for d in (RESULTS_DIR, FIG_DIR, MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)

    fracs, seeds = (FRACS, SEEDS) if not args.fast else ([0.10, 1.00], [0])
    fracs, seeds = (args.fracs or fracs), (args.seeds if args.seeds is not None else seeds)
    df = load_dataset(labeled_only=False)
    if args.fast:
        keep = df.sample(frac=1.0, random_state=0).groupby("time_step").head(120).index
        df = df.loc[np.sort(keep)].reset_index(drop=True)
    cols_b = feature_cols(df, "time")
    cols_notp = [c for c in cols_b if c not in TIME_PROXY]
    print(f"Setting B: {len(cols_b)} features ({len(cols_notp)} without time proxy). Rows {len(df)}", flush=True)

    val = df[df["time_step"].isin(list(VAL_STEPS)) & df["y"].notna()]
    test = df[df["time_step"].isin(list(TEST_STEPS)) & df["y"].notna()]
    y_val, y_test = val["y"].to_numpy(int), test["y"].to_numpy(int)
    t_test = test["time_step"].to_numpy()

    runs_csv = RESULTS_DIR / "e7_runs.csv"
    done = set()
    if runs_csv.exists() and runs_csv.stat().st_size > 0 and not args.fast:
        prev = pd.read_csv(runs_csv)
        done = {(r.method, float(r.frac), int(r.seed)) for r in prev.itertuples()}
    elif runs_csv.exists():
        runs_csv.unlink()

    splits = {}
    for key, steps in [("fit30", TRAIN_STEPS), ("fit34", list(TRAIN_STEPS) + list(VAL_STEPS))]:
        for fs, cols in [("time", cols_b), ("notp", cols_notp)]:
            splits[(key, fs)] = _fit_split(df, steps, cols)

    for method in args.methods:
        fs = "notp" if method == "sup_random_notp" else "time"
        cols = cols_notp if fs == "notp" else cols_b
        Xv, Xt = val[cols].to_numpy(np.float32), test[cols].to_numpy(np.float32)
        for frac in fracs:
            if method in SMALL_ONLY and frac > 0.10:
                continue
            for seed in seeds:
                if (method, float(frac), seed) in done:
                    print(f"[{method}] frac={frac} seed={seed} (cached)", flush=True)
                    continue
                t0 = time.time()
                # 1) fit 1-30 -> threshold on val 31-34
                X30, pool30, y30, unk30 = splits[("fit30", fs)]
                b30 = max(MIN_LABELS, int(round(frac * len(pool30))))
                emb_cache = {}
                m30, _, _ = train_method(method, X30, pool30, y30, unk30, b30, seed, emb_cache)
                p_val = m30.predict_proba(Xv)[:, 1]
                thr = tune_threshold(y_val, p_val)
                val_pr = float(average_precision_score(y_val, p_val))
                # 2) refit 1-34 -> test 35-49
                X34, pool34, y34, unk34 = splits[("fit34", fs)]
                b34 = max(MIN_LABELS, int(round(frac * len(pool34))))
                emb_cache = {}
                m34, n_lab, n_ill = train_method(method, X34, pool34, y34, unk34, b34, seed, emb_cache)
                p_test = m34.predict_proba(Xt)[:, 1]
                met = evaluate(y_test, p_test, thr)
                row = {"method": method, "frac": frac, "seed": seed, "n_labels": n_lab,
                       "n_illicit_labels": n_ill, "val_pr_auc": val_pr,
                       **{k: met[k] for k in ("threshold", "precision", "recall", "f1", "pr_auc")},
                       **_period_metrics(y_test, p_test, t_test, thr),
                       "fit_seconds": round(time.time() - t0, 1)}
                new = not runs_csv.exists() or runs_csv.stat().st_size == 0
                with runs_csv.open("a", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=RUN_FIELDS)
                    if new:
                        w.writeheader()
                    w.writerow(row)
                print(f"[{method}] frac={frac} seed={seed} labels={n_lab} (illicit {n_ill}) "
                      f"val_pr={val_pr:.3f} test_pr={met['pr_auc']:.3f} f1={met['f1']:.3f} "
                      f"43-49 pr={row['pr_auc_43_49']:.3f} ({row['fit_seconds']}s)", flush=True)
                if seed == 0 and not args.fast:
                    try:
                        joblib.dump(m34, MODEL_DIR / f"e7_{method}_{frac:g}.joblib")
                    except Exception as e:  # noqa: BLE001
                        print(f"WARNING: model save failed for {method} {frac}: {e}", flush=True)

    runs = pd.read_csv(runs_csv)
    summarize(runs, test_prevalence=float(y_test.mean()), fast=args.fast)
    return 0


def summarize(runs, test_prevalence, fast=False):
    # team results.csv: one row per (method, frac, seed)
    clear_results("E7")
    for r in runs.itertuples():
        fs = "time-tp" if r.method == "sup_random_notp" else "time"
        note = (f"frac={r.frac:g};n_labels={r.n_labels};n_illicit_labels={r.n_illicit_labels};"
                f"pr_auc_35_42={r.pr_auc_35_42:.4f};pr_auc_43_49={r.pr_auc_43_49:.4f};threshold=tuned_on_val")
        log_result("E7", model=f"{LABELS[r.method]}@{r.frac:g}", feature_set=fs, seed=int(r.seed),
                   metrics={k: float(getattr(r, k)) for k in ("precision", "recall", "f1", "pr_auc", "threshold")},
                   note=note)

    agg_cols = ["n_labels", "n_illicit_labels", "val_pr_auc", "precision", "recall", "f1", "pr_auc",
                "pr_auc_35_42", "pr_auc_43_49", "f1_35_42", "f1_43_49", "fit_seconds"]
    g = runs.groupby(["method", "frac"])[agg_cols]
    summary = g.mean().add_suffix("_mean").join(g.std().add_suffix("_std")).reset_index()
    summary.insert(1, "label", summary["method"].map(LABELS))
    summary.to_csv(RESULTS_DIR / "e7_summary.csv", index=False, encoding="utf-8-sig")
    print(f"Saved results/e7_summary.csv ({len(summary)} rows)", flush=True)

    unsup = None
    e6b_b = RESULTS_DIR / "e6b_B_summary.csv"
    if e6b_b.exists():
        s = pd.read_csv(e6b_b)
        s = s[s["mode"] == "best"] if "mode" in s.columns else s
        if len(s):
            # per model, the configuration chosen on validation (not the extra comparison preps);
            # then the strongest of those models on test
            chosen = s.loc[s.groupby("model")["val_pr_auc"].idxmax()]
            best = chosen.loc[chosen["pr_auc_mean"].idxmax()]
            unsup = (float(best["pr_auc_mean"]), f"{best['model']}")

    def curve(metric, fname, ylabel, title):
        fig, ax = plt.subplots(figsize=(8.5, 5))
        for m in METHODS:
            d = summary[summary["method"] == m].sort_values("frac")
            if len(d) == 0:
                continue
            ax.errorbar(d["frac"] * 100, d[f"{metric}_mean"], yerr=d[f"{metric}_std"].fillna(0),
                        marker="o", capsize=3, label=LABELS[m])
        if metric == "pr_auc":
            ax.axhline(test_prevalence, color="gray", ls=":", label=f"Random guess ({test_prevalence:.3f})")
            if unsup:
                ax.axhline(unsup[0], color="black", ls="--", label=f"Best unsupervised, {unsup[1]} ({unsup[0]:.3f})")
        ax.set_xscale("log")
        ax.set_xticks([1, 5, 10, 50, 100])
        ax.set_xticklabels(["1%", "5%", "10%", "50%", "100%"])
        ax.set(xlabel="Share of training labels used (log scale)", ylabel=ylabel, title=title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5, loc="lower right")
        fig.tight_layout()
        fig.savefig(FIG_DIR / fname, dpi=130)
        plt.close(fig)

    curve("pr_auc", "e7_label_curve_pr_auc.png", "Test PR-AUC (illicit)",
          "E7: test PR-AUC by label share (setting B, seeds 0-2)")
    curve("f1", "e7_label_curve_f1.png", "Test F1 (illicit, threshold tuned on 31-34)",
          "E7: test F1 by label share (setting B, seeds 0-2)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, per in zip(axes, ["35_42", "43_49"]):
        for m in METHODS:
            d = summary[summary["method"] == m].sort_values("frac")
            if len(d):
                ax.plot(d["frac"] * 100, d[f"pr_auc_{per}_mean"], marker="o", label=LABELS[m])
        ax.set_xscale("log")
        ax.set_xticks([1, 5, 10, 50, 100])
        ax.set_xticklabels(["1%", "5%", "10%", "50%", "100%"])
        ax.set(title=f"Test steps {per.replace('_', '-')}", xlabel="Share of training labels used")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Test PR-AUC (illicit)")
    axes[1].legend(fontsize=7.5)
    fig.suptitle("E7: before (35-42) vs after (43-49) the dark market shutdown")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e7_period.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
