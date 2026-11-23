"""E8b: does semi-supervised learning speed up recovery after the step-43 shift? (setting B)

E8 showed that a few labels from steps 43-45 recover PR-AUC on 46-49, with all labels of 1-42.
Here the old period is label-scarce too (10% of the labelled rows of 1-42, as in E7), and a
share g of the labelled rows of 43-45 is added (g = 0, 0.1, 0.25, 0.5). All transactions of
1-45 are available WITHOUT labels. Test: labelled rows of 46-49 (116 illicit / 2,505).

Methods (same as E7/E7c):
  LGBM   LightGBM (team E2 parameters) on the revealed labels only
  ST     self-training (team, threshold 0.95, 5 rounds); unlabelled pool = rows of 1-45 without
         a revealed label, capped at 30,000 with the rows of 43-45 kept first (up to 15,000)
  LS     label spreading (E7, quantile + PCA-20 embedding fitted on all rows of 1-45)
  ENS    mean probability of LGBM + ST + LS (a rank average over 46-49 would use future steps)
Seeds 0, 1, 2 (label selection and model). Metric: PR-AUC on 46-49.
Outputs: results/e8b_runs.csv, e8b_summary.csv, figures/e8b_shift_ssl.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from common import BASE
from common import feature_cols, load_dataset
import e7_semi_supervised as e7
import e7c_search as e7c
from e7d_test_time_st import self_train

R = BASE / "results"
F = BASE / "figures"
OLD_FRAC = 0.10
POST_FRACS = [0.0, 0.1, 0.25, 0.5]
SEEDS = [0, 1, 2]


def main(seeds=SEEDS, post_fracs=POST_FRACS):
    df = load_dataset(labeled_only=False)
    X = df[feature_cols(df, "time")].to_numpy(np.float32)
    t = df["time_step"].to_numpy()
    y = df["y"].to_numpy()
    fit = np.flatnonzero(t <= 45)
    test = np.flatnonzero((t >= 46) & pd.notna(y))
    y_test = y[test].astype(int)
    old_pool = np.flatnonzero((t <= 42) & pd.notna(y))
    post_pool = np.flatnonzero((t >= 43) & (t <= 45) & pd.notna(y))

    Xf = X[fit]
    emb = e7.Embedding(seed=0).fit(Xf)
    ls_cache = {"emb": emb, "Z": emb.transform(Xf)}
    rows = []
    for seed in seeds:
        rng = np.random.RandomState(seed)
        old_lab = old_pool[rng.choice(len(old_pool), int(round(OLD_FRAC * len(old_pool))), replace=False)]
        for g in post_fracs:
            post_lab = post_pool[np.random.RandomState(seed + 50).choice(
                len(post_pool), int(round(g * len(post_pool))), replace=False)] if g > 0 else np.array([], int)
            lab = np.concatenate([old_lab, post_lab])
            yL = y[lab].astype(int)
            unl = np.setdiff1d(fit, lab)
            new_unl = unl[t[unl] >= 43]
            new_unl = new_unl[np.random.RandomState(seed + 7).permutation(len(new_unl))][:15000]
            old_unl = np.setdiff1d(unl, new_unl)
            old_unl = old_unl[np.random.RandomState(seed + 8).permutation(len(old_unl))][:30000 - len(new_unl)]
            U = np.concatenate([new_unl, old_unl])

            p = {}
            p["LGBM"] = e7c.lgbm("team", seed).fit(X[lab], yL).predict_proba(X[test])[:, 1]
            p["ST"] = self_train(X[lab], yL, X[U], seed).predict_proba(X[test])[:, 1]
            # label spreading on local indices of `fit`; revealed labels passed as the pool with budget = all
            lab_l = np.searchsorted(fit, lab)
            unl_l = np.searchsorted(fit, np.setdiff1d(fit, lab))
            ls, _, _ = e7.train_method("label_spreading", Xf, lab_l, yL, unl_l, len(lab), seed, dict(ls_cache))
            p["LS"] = ls.predict_proba(X[test])[:, 1]
            p["ENS"] = np.mean([p[m] for m in ("LGBM", "ST", "LS")], axis=0)
            for m, pr in p.items():
                rows.append({"method": m, "post_frac": g, "seed": seed, "n_old_labels": len(old_lab),
                             "n_post_labels": len(post_lab), "n_post_illicit": int(y[post_lab].sum()) if len(post_lab) else 0,
                             "pr_auc_46_49": average_precision_score(y_test, pr)})
            print(f"seed={seed} g={g}: " + " ".join(f"{m}={rows[-4 + i]['pr_auc_46_49']:.3f}" for i, m in enumerate(p)),
                  flush=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(R / "e8b_runs.csv", index=False)
    summ = runs.groupby(["method", "post_frac"]).agg(pr_auc_mean=("pr_auc_46_49", "mean"),
                                                     pr_auc_std=("pr_auc_46_49", "std"),
                                                     n_post_illicit=("n_post_illicit", "mean")).reset_index()
    summ.to_csv(R / "e8b_summary.csv", index=False, encoding="utf-8-sig")
    print(summ.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for m in ("LGBM", "ST", "LS", "ENS"):
        d = summ[summ.method == m]
        ax.errorbar(d.post_frac * 100, d.pr_auc_mean, yerr=d.pr_auc_std, marker="o", capsize=3, label=m)
    ax.axhline(y_test.mean(), color="gray", ls=":", label=f"Random guess ({y_test.mean():.3f})")
    ax.set(xlabel="Share of labelled rows of steps 43-45 revealed (%)", ylabel="PR-AUC on steps 46-49",
           title="E8b: recovery after the shutdown with 10% of old labels (setting B, seeds 0-2)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(F / "e8b_shift_ssl.png", dpi=130); plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
