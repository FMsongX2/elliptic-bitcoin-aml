"""E7b: E7 (semi-supervised, setting B) with two additions. Training code is reused from E7.

1. Distance space for the distance-based methods (label_spreading, cluster_rep, cluster_prop):
   quantile_57   QuantileTransformer + PCA-20 on 57 features (the E7 embedding, e7.Embedding)
   quantile_54   same without the time proxy agg_6/7/9
   yeojohnson_54 Yeo-Johnson + PCA-20 on 54 features
   Chosen from prep_research_b.py (validation only): quantile_54 best for kNN at 1-10% labels,
   yeojohnson_54 best for Label Spreading at 10%.
2. Realistic threshold. E7 tuned the threshold on ALL labeled validation rows even when only
   1% of training labels were revealed. E7b also tunes it on the same share of validation labels
   (random subset, at least 20 rows; 0.5 if the subset has no illicit row) and reports both.
   PR-AUC does not depend on the threshold.

Protocol otherwise identical to E7 (fit 1-30 -> threshold on 31-34, refit 1-34 -> test 35-49,
seeds 0-2, UNLABELED_CAP 60000). Outputs: results/e7b_runs.csv (resumable), e7b_summary.csv,
results.csv rows "E7b", figures/e7b_*.png.
"""
import argparse
import csv
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from common import BASE
import common
from common import TEST_STEPS, TRAIN_STEPS, VAL_STEPS, clear_results, feature_cols, load_dataset, log_result, tune_threshold
import e7_semi_supervised as e7
from prep_research_b import Pipe

RESULTS_DIR = BASE / "results"
FIG_DIR = BASE / "figures"
DIST_METHODS = ["label_spreading", "cluster_rep", "cluster_prop"]
EMBEDDINGS = ["quantile_57", "quantile_54", "yeojohnson_54"]
OTHER_METHODS = ["sup_random", "sup_random_notp", "self_training", "active_unc"]
MIN_VAL_LABELS = 20
FIELDS = ["method", "embedding", "frac", "seed", "n_labels", "n_illicit_labels", "val_pr_auc", "pr_auc",
          "pr_auc_35_42", "pr_auc_43_49",
          "threshold_full", "precision_full", "recall_full", "f1_full",
          "n_val_labels_real", "threshold_real", "precision_real", "recall_real", "f1_real", "fit_seconds"]


def make_embedding(name, X, cols):
    if name == "quantile_57":
        emb = e7.Embedding(seed=0).fit(X)
    else:
        emb = Pipe(name, cols, seed=0).fit(X)
    return {"emb": emb, "Z": emb.transform(X)}


def realistic_threshold(y_val, p_val, frac, seed):
    """Tune on a random `frac` share of the validation labels (>= MIN_VAL_LABELS rows)."""
    rng = np.random.RandomState(seed + 100)
    n = min(len(y_val), max(MIN_VAL_LABELS, int(round(frac * len(y_val)))))
    sub = rng.choice(len(y_val), n, replace=False)
    if y_val[sub].sum() == 0:
        return 0.5, n
    return tune_threshold(y_val[sub], p_val[sub]), n


def _prf(y, p, thr):
    pred = p >= thr
    return (precision_score(y, pred, zero_division=0), recall_score(y, pred, zero_division=0),
            f1_score(y, pred, zero_division=0))


def configs(methods, embeddings):
    out = []
    for m in methods:
        if m in DIST_METHODS:
            out += [(m, e) for e in embeddings]
        else:
            out.append((m, "-"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="*", default=OTHER_METHODS + DIST_METHODS)
    ap.add_argument("--embeddings", nargs="*", default=EMBEDDINGS)
    ap.add_argument("--fracs", nargs="*", type=float, default=e7.FRACS)
    ap.add_argument("--seeds", nargs="*", type=int, default=e7.SEEDS)
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args(argv)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(labeled_only=False)
    if args.fast:
        keep = df.sample(frac=1.0, random_state=0).groupby("time_step").head(120).index
        df = df.loc[np.sort(keep)].reset_index(drop=True)
    cols_b = feature_cols(df, "time")
    cols_notp = [c for c in cols_b if c not in e7.TIME_PROXY]
    val = df[df["time_step"].isin(list(VAL_STEPS)) & df["y"].notna()]
    test = df[df["time_step"].isin(list(TEST_STEPS)) & df["y"].notna()]
    y_val, y_test = val["y"].to_numpy(int), test["y"].to_numpy(int)
    t_test = test["time_step"].to_numpy()

    runs_csv = RESULTS_DIR / "e7b_runs.csv"
    done = set()
    if runs_csv.exists() and runs_csv.stat().st_size > 0 and not args.fast:
        done = {(r.method, r.embedding, float(r.frac), int(r.seed)) for r in pd.read_csv(runs_csv).itertuples()}
    elif runs_csv.exists():
        runs_csv.unlink()

    splits = {}
    for key, steps in [("fit30", TRAIN_STEPS), ("fit34", list(TRAIN_STEPS) + list(VAL_STEPS))]:
        for fs, cols in [("time", cols_b), ("notp", cols_notp)]:
            splits[(key, fs)] = e7._fit_split(df, steps, cols)
    emb_store = {}  # (fit key, embedding) -> {"emb", "Z"}; embeddings use no labels, shared across seeds

    for method, emb_name in configs(args.methods, args.embeddings):
        fs = "notp" if method == "sup_random_notp" else "time"
        cols = cols_notp if fs == "notp" else cols_b
        Xv, Xt = val[cols].to_numpy(np.float32), test[cols].to_numpy(np.float32)
        for frac in args.fracs:
            if method in e7.SMALL_ONLY and frac > 0.10:
                continue
            for seed in args.seeds:
                if (method, emb_name, float(frac), seed) in done:
                    continue
                t0 = time.time()
                models = {}
                for key in ("fit30", "fit34"):
                    X, pool, y, unk = splits[(key, fs)]
                    budget = max(e7.MIN_LABELS, int(round(frac * len(pool))))
                    cache = {}
                    if method in DIST_METHODS:
                        if (key, emb_name) not in emb_store:
                            emb_store[(key, emb_name)] = make_embedding(emb_name, X, cols)
                        cache = dict(emb_store[(key, emb_name)])
                    models[key] = e7.train_method(method, X, pool, y, unk, budget, seed, cache)
                p_val = models["fit30"][0].predict_proba(Xv)[:, 1]
                m34, n_lab, n_ill = models["fit34"]
                p_test = m34.predict_proba(Xt)[:, 1]
                thr_full = tune_threshold(y_val, p_val)
                thr_real, n_val_real = realistic_threshold(y_val, p_val, frac, seed)
                pf, rf, ff = _prf(y_test, p_test, thr_full)
                pr_, rr_, fr_ = _prf(y_test, p_test, thr_real)
                per = e7._period_metrics(y_test, p_test, t_test, thr_full)
                row = {"method": method, "embedding": emb_name, "frac": frac, "seed": seed, "n_labels": n_lab,
                       "n_illicit_labels": n_ill, "val_pr_auc": average_precision_score(y_val, p_val),
                       "pr_auc": average_precision_score(y_test, p_test),
                       "pr_auc_35_42": per["pr_auc_35_42"], "pr_auc_43_49": per["pr_auc_43_49"],
                       "threshold_full": thr_full, "precision_full": pf, "recall_full": rf, "f1_full": ff,
                       "n_val_labels_real": n_val_real, "threshold_real": thr_real,
                       "precision_real": pr_, "recall_real": rr_, "f1_real": fr_,
                       "fit_seconds": round(time.time() - t0, 1)}
                new = not runs_csv.exists() or runs_csv.stat().st_size == 0
                with runs_csv.open("a", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=FIELDS)
                    if new:
                        w.writeheader()
                    w.writerow(row)
                print(f"[{method}|{emb_name}] frac={frac} seed={seed} test_pr={row['pr_auc']:.3f} "
                      f"f1_full={ff:.3f} f1_real={fr_:.3f} ({row['fit_seconds']}s)", flush=True)

    summarize(pd.read_csv(runs_csv), test_prevalence=float(y_test.mean()))
    return 0


def summarize(runs, test_prevalence):
    clear_results("E7b")
    for r in runs.itertuples():
        # cluster methods use the 54 columns only to pick representatives; their classifier gets all 57
        fs = "time-tp" if (r.method == "sup_random_notp"
                           or (r.method == "label_spreading" and str(r.embedding).endswith("_54"))) else "time"
        label = e7.LABELS[r.method] + (f" [{r.embedding}]" if r.embedding != "-" else "")
        note = (f"frac={r.frac:g};embedding={r.embedding};threshold=tuned_on_all_val;"
                f"f1_real={r.f1_real:.4f};precision_real={r.precision_real:.4f};recall_real={r.recall_real:.4f};"
                f"threshold_real={r.threshold_real:.2f};n_val_labels_real={r.n_val_labels_real};"
                f"pr_auc_35_42={r.pr_auc_35_42:.4f};pr_auc_43_49={r.pr_auc_43_49:.4f}")
        log_result("E7b", model=f"{label}@{r.frac:g}", feature_set=fs, seed=int(r.seed),
                   metrics={"precision": r.precision_full, "recall": r.recall_full, "f1": r.f1_full,
                            "pr_auc": r.pr_auc, "threshold": r.threshold_full}, note=note)
    num = ["n_labels", "n_illicit_labels", "val_pr_auc", "pr_auc", "pr_auc_35_42", "pr_auc_43_49",
           "precision_full", "recall_full", "f1_full", "threshold_real", "precision_real", "recall_real", "f1_real",
           "fit_seconds"]
    g = runs.groupby(["method", "embedding", "frac"])[num]
    summ = g.mean().add_suffix("_mean").join(g.std().add_suffix("_std")).reset_index()
    summ.to_csv(RESULTS_DIR / "e7b_summary.csv", index=False, encoding="utf-8-sig")
    print(f"Saved results/e7b_summary.csv ({len(summ)} rows)", flush=True)

    ticks = ([1, 5, 10, 50, 100], ["1%", "5%", "10%", "50%", "100%"])
    # Figure: distance space per distance-based method
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    for ax, m in zip(axes, DIST_METHODS):
        for emb in EMBEDDINGS:
            d = summ[(summ.method == m) & (summ.embedding == emb)].sort_values("frac")
            if len(d):
                ax.errorbar(d.frac * 100, d.pr_auc_mean, yerr=d.pr_auc_std.fillna(0), marker="o", capsize=3, label=emb)
        ax.set_xscale("log"); ax.set_xticks(ticks[0]); ax.set_xticklabels(ticks[1])
        ax.set(title=e7.LABELS[m], xlabel="Share of training labels used"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("Test PR-AUC (illicit)"); axes[0].legend(fontsize=8)
    fig.suptitle("E7b: distance space for distance-based semi-supervised methods (setting B, seeds 0-2)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "e7b_embedding.png", dpi=130); plt.close(fig)

    # Figure: F1 with threshold tuned on all val labels vs on the same label share
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    best = summ[(summ.embedding.isin(["-", "quantile_54"]))]
    for ax, kind, title in [(axes[0], "full", "Threshold tuned on all val labels (E7)"),
                            (axes[1], "real", "Threshold tuned on the same share of val labels")]:
        for (m, emb), d in best.groupby(["method", "embedding"]):
            d = d.sort_values("frac")
            ax.plot(d.frac * 100, d[f"f1_{kind}_mean"], marker="o",
                    label=e7.LABELS[m] + (f" [{emb}]" if emb != "-" else ""))
        ax.set_xscale("log"); ax.set_xticks(ticks[0]); ax.set_xticklabels(ticks[1])
        ax.set(title=title, xlabel="Share of labels used (train and val)"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("Test F1 (illicit)"); axes[1].legend(fontsize=7)
    fig.suptitle("E7b: how much of the F1 came from fully labeled validation")
    fig.tight_layout(); fig.savefig(FIG_DIR / "e7b_threshold.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
