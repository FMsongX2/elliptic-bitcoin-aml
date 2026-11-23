"""E9: learn the illicit patterns without relying on the dominant pre-43 cue, using no time
information (54 features = setting B without agg_6/7/9; time_step is never a feature).

Diagnosis (minority_analysis.py): the pre-43 rule "single input -> suspicious" (illicit rate
21.9% among single-input rows in 1-34) collapses after step 43 (0.6%), while multi-input illicit
(a ~5% minority before) becomes the main pattern. A global model then ranks single-input licit
rows above the real illicit ones.

Methods (all trained on every labelled row of the training steps, LightGBM team E2 parameters):
  M0 global      one model, all 54 features (baseline)
  M1 group_rank  M0 scores turned into percentiles WITHIN each group (single / multi input) of the
                 scored batch, so neither group can monopolise the top of the ranking
  M2 group_em    M0 probabilities re-weighted per group with the class prior of the scored batch,
                 estimated without labels by EM (Saerens, Latinne & Decaestecker 2002). The batch is
                 the transactions of one time step (all rows, labels unused), which is what a
                 deployed scorer receives every ~2 weeks; nothing about the shutdown is assumed.
  M2c group_em_cal   M2 on probabilities calibrated by 5-fold Platt scaling on the training rows
                      (CalibratedClassifierCV), since EM assumes calibrated posteriors
                      (Alexandari et al. 2020)
  M2s group_em_shrink M2 with the estimated prior pulled halfway (geometric mean) to the training
                      prior, to stop the estimate collapsing to 0
  M2r group_em_rel    M2c, but the change is measured against a reference prior estimated the
                      same way (EM on ALL rows of each of the last 10 training steps, out-of-fold
                      calibrated probabilities for labelled rows), so the batch and the reference
                      describe the same population; unchanged batches are left (almost) unchanged
  M3 no_input    M0 without the features whose dominant value marks single-input transactions
                 (agreement with the single-input flag >= 0.9 on the training rows)
Group = whether local_3 (number of inputs) equals its most frequent training value.
Train 1-34 -> test 35-49 (also 1-42 -> 43-49). Seeds 0, 1, 2. Metric: PR-AUC pooled over periods.
Outputs: results/e9_runs.csv, e9_summary.csv
"""
import json

import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

from common import BASE
from common import feature_cols, load_dataset

R = BASE / "results"
TIME_PROXY = ["agg_6", "agg_7", "agg_9"]


def lgbm(seed):
    return lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.5, random_state=seed, n_jobs=-1, verbose=-1)


def em_prior(p, prior_train, iters=100, tol=1e-6):
    """Saerens et al. 2002: estimate the class-1 prior of a batch and return adjusted probabilities."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    pi = prior_train
    for _ in range(iters):
        a = (pi / prior_train) * p
        b = ((1 - pi) / (1 - prior_train)) * (1 - p)
        q = a / (a + b)
        new = q.mean()
        if abs(new - pi) < tol:
            break
        pi = new
    return q, pi


def reweight(p, prior_train, pi):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    a = (pi / prior_train) * p
    return a / (a + ((1 - pi) / (1 - prior_train)) * (1 - p))


def ap(y, p):
    return float(average_precision_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def main():
    df = load_dataset(labeled_only=False)
    cols = [c for c in feature_cols(df, "time") if c not in TIME_PROXY]
    X = df[cols].to_numpy(np.float32)
    t = df.time_step.to_numpy()
    y = df.y.to_numpy()
    lab = pd.notna(y)
    mode3 = df.loc[t <= 30, "local_3"].value_counts().index[0]
    single = df.local_3.to_numpy() == mode3

    # features that encode the single-input flag (decided on training rows only)
    tr_rows = t <= 34
    drop = []
    for j, c in enumerate(cols):
        v = X[tr_rows, j]
        vals, cnt = np.unique(v, return_counts=True)
        flag = v == vals[cnt.argmax()]
        if cnt.max() / len(v) > 0.3 and (flag == single[tr_rows]).mean() >= 0.9:
            drop.append(c)
    keep = np.array([c not in drop for c in cols])
    print(f"M3 drops {len(drop)} features: {drop}", flush=True)

    rows = []
    for train_end, eval_steps in [(34, range(35, 50)), (42, range(43, 50))]:
        trm = (t <= train_end) & lab
        y_tr = y[trm].astype(int)
        prior = {g: y_tr[single[trm] == g].mean() for g in (True, False)}
        for seed in (0, 1, 2):
            m_all = lgbm(seed).fit(X[trm], y_tr)
            m_noin = lgbm(seed).fit(X[trm][:, keep], y_tr)
            m_cal = CalibratedClassifierCV(lgbm(seed), method="sigmoid", cv=5).fit(X[trm], y_tr)
            # reference prior per group on the SAME population as the scored batches (all rows of a
            # step, labelled or not), from the last 10 training steps. Labelled rows get out-of-fold
            # calibrated probabilities (5-fold), unlabelled rows the full calibrated model.
            lab_idx = np.flatnonzero(trm)
            oof = np.zeros(len(lab_idx))
            for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(lab_idx, y_tr):
                cm = CalibratedClassifierCV(lgbm(seed), method="sigmoid", cv=3).fit(X[lab_idx[a]], y_tr[a])
                oof[b] = cm.predict_proba(X[lab_idx[b]])[:, 1]
            ref_rows = np.flatnonzero((t <= train_end) & (t > train_end - 10))
            p_ref = m_cal.predict_proba(X[ref_rows])[:, 1]
            pos = {r: i for i, r in enumerate(lab_idx)}
            for k, r in enumerate(ref_rows):
                if r in pos:
                    p_ref[k] = oof[pos[r]]
            ref = {}
            for gv in (True, False):
                ests = []
                for s in sorted(set(t[ref_rows])):
                    mk = (t[ref_rows] == s) & (single[ref_rows] == gv)
                    ests.append(em_prior(p_ref[mk], prior[gv])[1])
                ref[gv] = float(np.mean(ests))
            print(f"  reference priors (all rows, steps {train_end - 9}-{train_end}): "
                  f"single {ref[True]:.4f}, multi {ref[False]:.4f}", flush=True)
            scores = {k: np.full(len(y), np.nan)
                      for k in ("global", "group_rank", "group_em", "group_em_cal", "group_em_shrink", "group_em_rel",
                                "no_input")}
            est_prior = []
            for s in eval_steps:
                batch = t == s                                   # all rows of the step, labels unused
                p = m_all.predict_proba(X[batch])[:, 1]
                pc = m_cal.predict_proba(X[batch])[:, 1]
                scores["global"][batch] = p
                scores["no_input"][batch] = m_noin.predict_proba(X[batch][:, keep])[:, 1]
                g = single[batch]
                r = np.empty(len(p))
                q = np.empty(len(p))
                qc = np.empty(len(p))
                qs = np.empty(len(p))
                qr = np.empty(len(p))
                for gv in (True, False):
                    gm = g == gv
                    r[gm] = rankdata(p[gm]) / gm.sum()
                    q[gm], pi = em_prior(p[gm], prior[gv])
                    qc[gm], pi_c = em_prior(pc[gm], prior[gv])
                    qs[gm] = reweight(p[gm], prior[gv], np.sqrt(max(pi, 1e-6) * prior[gv]))
                    # relative change vs the reference prior of the same population
                    pi_eff = min(0.99, prior[gv] * pi_c / max(ref[gv], 1e-6))
                    qr[gm] = reweight(pc[gm], prior[gv], pi_eff)
                    est_prior.append({"step": s, "group": "single" if gv else "multi", "prior_train": prior[gv],
                                      "prior_est": pi, "prior_est_cal": pi_c, "prior_true_labelled": float(np.nanmean(y[batch][gm]))})
                scores["group_rank"][batch] = r
                scores["group_em"][batch] = q
                scores["group_em_cal"][batch] = qc
                scores["group_em_shrink"][batch] = qs
                scores["group_em_rel"][batch] = qr
            for method, sc in scores.items():
                for per, pm in [("35-42", (t >= 35) & (t <= 42)), ("43-49", t >= 43), ("35-49", t >= 35)]:
                    e = pm & lab & np.isin(t, list(eval_steps))
                    if e.sum() == 0 or (train_end == 42 and per != "43-49"):
                        continue
                    rows.append({"train": f"1-{train_end}", "method": method, "seed": seed, "eval": per,
                                 "pr_auc": ap(y[e].astype(int), sc[e]),
                                 "pr_auc_multi": ap(y[e & ~single].astype(int), sc[e & ~single])})
            if seed == 0:
                pd.DataFrame(est_prior).to_csv(R / f"e9_em_priors_train1-{train_end}.csv", index=False)
            print(f"train 1-{train_end} seed {seed} done", flush=True)

    runs = pd.DataFrame(rows)
    runs.to_csv(R / "e9_runs.csv", index=False)
    summ = runs.groupby(["train", "eval", "method"]).agg(pr_auc=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
                                                        pr_auc_multi=("pr_auc_multi", "mean")).reset_index()
    summ.to_csv(R / "e9_summary.csv", index=False, encoding="utf-8-sig")
    (R / "e9_dropped_features.json").write_text(json.dumps(drop), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
