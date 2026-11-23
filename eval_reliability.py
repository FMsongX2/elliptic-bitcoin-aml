"""Reliability and operational evaluation of the final setting-B methods (test steps 35-49).

Uses stored test predictions (no new model selection):
  e7c_preds.joblib          seeds 0-2: self-training (ST), LightGBM (LGBM), label spreading (LS)
  e7c_confirm_preds.joblib  seeds 3-5: the same three methods
  recomputed here           LightGBM with 100% of the labels (seeds 0-5) and the unsupervised
                            BIRCH detector of E6b-B (std_notp, threshold 1.0, 8 clusters, PCA-20,
                            direction -1; seeds 0-2)
Ensembles of ST + LGBM + LS. A rank average over the whole test period lets a step see the score
distribution of later steps, so two rules without that are reported (both, none is picked):
  ENS-past  at step t, rank average within the scored rows of steps 35..t
  ENS-prob  mean of the three probabilities (uses no test distribution at all)

1. Bootstrap 95% intervals of test PR-AUC (seed-averaged), two ways:
     rows   resample labelled test transactions within each time step (1,000 times)
     steps  resample whole time steps (15 blocks) with replacement (1,000 times); reflects the
            step-to-step variation. Neither covers selection on the test period.
   plus paired intervals for differences (same resample for both methods) and the seed-level
   standard error of each difference.
2. Top-k: the k highest scores among the LABELLED transactions of each time step (k = 50, 100,
   200); precision@k and recall@k per period. This is not a ranking of the full transaction stream.
3. Calibration of the probability outputs (LGBM, ST), per seed then averaged: Brier score, expected
   calibration error (10 equal-width bins), mean predicted probability vs observed illicit share.
Outputs: results/eval_bootstrap.csv, eval_paired.csv, eval_seed_ap.csv, eval_topk.csv,
eval_calibration.csv, eval_reliability_bins.json,
figures/eval_ci.png, eval_topk.png, eval_calibration.png
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from common import BASE
from common import load_dataset
import e7c_search as e7c
from detectors_e6b import BirchDetector

R = BASE / "results"
F = BASE / "figures"
N_BOOT = 1000
FRACS = [0.01, 0.05, 0.10]
CID = {
    "ST": e7c.config_id(dict(kind="st", base="team", half_life=None, unsup=False, thr=0.95, balanced=False, plw=1.0)),
    "LGBM": e7c.config_id(dict(kind="sup", base="team", half_life=None, unsup=False)),
    "LS": e7c.config_id(dict(kind="ls")),
}


def ensemble_scores(ps, steps, rule):
    """Combine member scores without using any time step later than the one being scored."""
    if rule == "prob":
        return np.mean(ps, axis=0)
    if rule != "past":
        raise ValueError(f"unknown rule {rule!r}: ranks over the whole test period use future steps")
    out = np.empty(len(steps))
    for s in np.unique(steps):
        ref = steps <= s
        now = steps[ref] == s
        out[steps == s] = np.mean([(rankdata(p[ref]) / ref.sum())[now] for p in ps], axis=0)
    return out


def boot_ap(y, p, weights):
    """Average precision under each bootstrap weight vector; equals sklearn on the resampled rows."""
    o = np.argsort(-p, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(p[o])), len(p) - 1]
    res = np.empty(len(weights))
    for i, w in enumerate(weights):
        wo = np.asarray(w)[o]
        tp = np.cumsum(wo * y[o], dtype=np.float64)[ends]
        seen = np.cumsum(wo, dtype=np.float64)[ends]
        prec = np.divide(tp, seen, out=np.zeros_like(tp), where=seen > 0)
        res[i] = np.sum(np.diff(np.r_[0.0, tp]) * prec) / tp[-1]
    return res


def collect(D):
    """Return y_test, steps, and {method: [pred per seed]}."""
    fit, pool, unk, ev = D.split(range(1, 35), range(35, 50))
    y = D.y[ev].astype(int)
    steps = D.t[ev]
    a = joblib.load(R / "e7c_preds.joblib")
    b = joblib.load(R / "e7c_confirm_preds.joblib")
    preds = {}
    for frac in FRACS:
        per_seed = {m: [] for m in ("ST", "LGBM", "LS")}
        for seed in range(6):
            for m in per_seed:
                p = a[(CID[m], "test", frac, seed)] if seed < 3 else b[(m, "test", frac, seed)]
                per_seed[m].append(np.asarray(p, dtype=np.float64))
        tag = f"{frac:.0%}"
        preds[f"LGBM {tag}"] = per_seed["LGBM"]
        preds[f"ST {tag}"] = per_seed["ST"]
        for rule in ("past", "prob"):
            preds[f"ENS-{rule} {tag}"] = [ensemble_scores([per_seed[m][s] for m in ("ST", "LGBM", "LS")], steps, rule)
                                          for s in range(6)]
    preds["LGBM 100%"] = [e7c.run(dict(kind="sup", base="team", half_life=None, unsup=False), D, e7c.TEST, 1.0, s)[0]
                          for s in range(6)]
    # unsupervised BIRCH (E6b-B config), fitted on all rows of 1-34 without labels
    Xs = D.X[fit][:, D.notp]
    sc = StandardScaler().fit(Xs)
    birch = []
    for s in range(3):
        det = BirchDetector(threshold=1.0, n_clusters=8, pca=20, seed=s).fit(sc.transform(Xs))
        birch.append(-det.score(sc.transform(D.X[ev][:, D.notp])))
    preds["BIRCH 0%"] = birch
    return y, steps, preds


def main():
    D = e7c.Data(load_dataset(labeled_only=False))
    y, steps, preds = collect(D)
    tags = [f"{f:.0%}" for f in FRACS]
    order = ["BIRCH 0%"] + [f"{m} {f}" for f in tags for m in ("LGBM", "ST", "ENS-past", "ENS-prob")] + ["LGBM 100%"]
    seed_ap = {m: np.array([average_precision_score(y, p) for p in preds[m]]) for m in order}
    point = {m: float(v.mean()) for m, v in seed_ap.items()}
    print({m: round(v, 3) for m, v in point.items()}, flush=True)

    pre = steps < 43
    pd.DataFrame([{"method": m, "seed": s, "pr_auc": seed_ap[m][s],
                   "pr_auc_35_42": average_precision_score(y[pre], p[pre]),
                   "pr_auc_43_49": average_precision_score(y[~pre], p[~pre])}
                  for m in order for s, p in enumerate(preds[m])]
                 ).to_csv(R / "eval_seed_ap.csv", index=False, encoding="utf-8-sig")

    rng = np.random.RandomState(0)
    step_ids = np.unique(steps)
    by_step = {s: np.flatnonzero(steps == s) for s in step_ids}
    weights = {"rows": [], "steps": []}
    count = lambda ix: np.bincount(ix, minlength=len(y)).astype(np.int32)
    for _ in range(N_BOOT):
        weights["rows"].append(count(np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in by_step.values()])))
        weights["steps"].append(count(np.concatenate([by_step[s] for s in rng.choice(step_ids, len(step_ids), replace=True)])))
    dist = {kind: {m: np.mean([boot_ap(y, p, weights[kind]) for p in preds[m]], axis=0) for m in order}
            for kind in weights}

    rows = []
    for m in order:
        r = {"method": m, "pr_auc": point[m]}
        for kind in weights:
            r[f"ci95_low_{kind}"], r[f"ci95_high_{kind}"] = np.percentile(dist[kind][m], [2.5, 97.5])
        rows.append(r)
    bt = pd.DataFrame(rows)
    bt.to_csv(R / "eval_bootstrap.csv", index=False, encoding="utf-8-sig")

    pairs = [(f"ENS-{rule} {f}", other) for rule in ("past", "prob") for f in tags
             for other in (f"ST {f}", f"LGBM {f}", "LGBM 100%")] + \
            [(f"ST {f}", f"LGBM {f}") for f in tags] + [("LGBM 1%", "BIRCH 0%")] + \
            [(f"ST {f}", "LGBM 100%") for f in tags] + [(f"LGBM {f}", "LGBM 100%") for f in tags]
    pr = []
    for a_, b_ in pairs:
        r = {"a": a_, "b": b_, "diff": point[a_] - point[b_]}
        for kind in weights:
            d = dist[kind][a_] - dist[kind][b_]
            r[f"ci95_low_{kind}"], r[f"ci95_high_{kind}"] = np.percentile(d, [2.5, 97.5])
            r[f"share_a_better_{kind}"] = float((d > 0).mean())
        if len(seed_ap[a_]) == len(seed_ap[b_]):
            r["seed_se"] = float((seed_ap[a_] - seed_ap[b_]).std(ddof=1) / np.sqrt(len(seed_ap[a_])))
        pr.append(r)
    pr = pd.DataFrame(pr)
    pr.to_csv(R / "eval_paired.csv", index=False, encoding="utf-8-sig")

    # 2. top-k per step, among labelled rows
    tk = []
    for m in order:
        for k in (50, 100, 200):
            for per, lo, hi in [("35-42", 35, 42), ("43-49", 43, 49)]:
                tp, flagged = [], 0
                for s in range(lo, hi + 1):
                    ix = by_step[s]
                    kk = min(k, len(ix))
                    tp.append(float(np.mean([y[ix[np.argsort(-p[ix])[:kk]]].sum() for p in preds[m]])))
                    flagged += kk
                n_ill = int(y[(steps >= lo) & (steps <= hi)].sum())
                tk.append({"method": m, "k": k, "period": per, "illicit_present": n_ill,
                           "illicit_caught": float(np.sum(tp)), "recall_at_k": float(np.sum(tp) / n_ill),
                           "precision_at_k": float(np.sum(tp) / flagged), "reviewed": flagged})
    tk = pd.DataFrame(tk)
    tk.to_csv(R / "eval_topk.csv", index=False, encoding="utf-8-sig")

    # 3. calibration of single runs, averaged over seeds
    cal = []
    rel = {}
    for m in ("LGBM 1%", "ST 1%", "LGBM 10%", "ST 10%", "LGBM 100%"):
        per = []
        for p in preds[m]:
            bins = np.clip((p * 10).astype(int), 0, 9)
            per.append({"brier": float(np.mean((p - y) ** 2)),
                        "ece": float(sum((bins == b).mean() * abs(p[bins == b].mean() - y[bins == b].mean())
                                         for b in range(10) if (bins == b).any())),
                        "mean_pred_35_42": float(p[pre].mean()), "mean_pred_43_49": float(p[~pre].mean())})
        cal.append({"method": m, **pd.DataFrame(per).mean().to_dict(),
                    "illicit_share_35_42": float(y[pre].mean()), "illicit_share_43_49": float(y[~pre].mean())})
        p, yy = np.concatenate(preds[m]), np.tile(y, len(preds[m]))
        bins = np.clip((p * 10).astype(int), 0, 9)
        rel[m] = [(float(p[bins == b].mean()), float(yy[bins == b].mean()), int((bins == b).sum()))
                  for b in range(10) if (bins == b).any()]
    cal = pd.DataFrame(cal)
    cal.to_csv(R / "eval_calibration.csv", index=False, encoding="utf-8-sig")
    (R / "eval_reliability_bins.json").write_text(json.dumps(rel, indent=1), encoding="utf-8")

    pd.set_option("display.width", 220)
    print(bt.round(3).to_string(index=False))
    print(pr.round(3).to_string(index=False))
    print(tk.round(3).to_string(index=False))
    print(cal.round(3).to_string(index=False))

    # figures
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    ys = np.arange(len(bt))[::-1]
    ax.errorbar(bt.pr_auc, ys, xerr=[bt.pr_auc - bt.ci95_low_steps, bt.ci95_high_steps - bt.pr_auc], fmt="none",
                ecolor="lightgray", elinewidth=6, label="95% CI, resampling time steps")
    ax.errorbar(bt.pr_auc, ys, xerr=[bt.pr_auc - bt.ci95_low_rows, bt.ci95_high_rows - bt.pr_auc], fmt="o",
                color="tab:blue", capsize=3, label="95% CI, resampling transactions")
    ax.set_yticks(ys); ax.set_yticklabels(bt.method)
    ax.axvline(y.mean(), color="gray", ls=":", label=f"Random guess ({y.mean():.3f})")
    ax.set(xlabel="Test PR-AUC (35-49, mean over seeds)", title="Bootstrap 95% intervals (setting B, no future ranks)")
    ax.grid(axis="x", alpha=0.3); ax.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout(); fig.savefig(F / "eval_ci.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    sel = ["BIRCH 0%", "LGBM 10%", "ST 10%", "ENS-past 10%", "ENS-prob 10%", "LGBM 100%"]
    for ax, per in zip(axes, ["35-42", "43-49"]):
        d = tk[(tk.period == per) & tk.method.isin(sel)]
        for m in sel:
            dd = d[d.method == m]
            ax.plot(dd.k, dd.recall_at_k, marker="o", label=m)
        ax.set(title=f"Test steps {per}", xlabel="Labelled transactions reviewed per time step (k)"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("Share of illicit transactions caught (recall@k)"); axes[1].legend(fontsize=8)
    fig.suptitle("Top-k among the labelled transactions of each 2-week step")
    fig.tight_layout(); fig.savefig(F / "eval_topk.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    for m, pts in rel.items():
        ax.plot([a for a, _, _ in pts], [b for _, b, _ in pts], marker="o", label=m)
    ax.plot([0, 1], [0, 1], color="gray", ls=":")
    ax.set(xlabel="Predicted probability (bin mean)", ylabel="Observed illicit share",
           title="Reliability on test 35-49 (single runs, pooled over seeds)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(F / "eval_calibration.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
