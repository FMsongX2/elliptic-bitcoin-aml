"""Why does a model trained before step 43 miss the post-43 illicit pattern, although that
pattern existed before as a ~5% minority? Time is excluded: 54 features (setting B without the
time proxies agg_6/7/9), no time_step, all labels of the training steps.

Subgroups by the dominant-value flag of local_3 (number of inputs; 1 input = 'single').
  A. Separability inside the training period (1-34): 5-fold CV PR-AUC of LightGBM on
     single-input rows only and on multi-input rows only.
  B. Transfer: train on 1-34 (or 1-42), evaluate on multi-input rows of 35-42 and 43-49, for
       global      one model on all rows
       specialist  one model on multi-input rows only
       upweight    global model with multi-input illicit rows weighted x10
       balanced    global model with every subgroup x class cell weighted to equal total weight
  C. Purity: in the illicit clusters that hold most post-43 illicit (shift_analysis), share of
     illicit among ALL labelled training rows assigned to those clusters.
Seeds 0-2. Outputs: results/minority_analysis_*.csv / .json
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer

from common import BASE
from common import feature_cols, load_dataset

R = BASE / "results"
TIME_PROXY = ["agg_6", "agg_7", "agg_9"]


def model(seed):
    return lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.5, random_state=seed, n_jobs=-1, verbose=-1)


def ap(y, p):
    return float(average_precision_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def main():
    df = load_dataset(labeled_only=True)
    cols = [c for c in feature_cols(df, "time") if c not in TIME_PROXY]
    X = df[cols].to_numpy(np.float32)
    y = df.y.to_numpy(int)
    t = df.time_step.to_numpy()
    mode = df.loc[t <= 30, "local_3"].value_counts().index[0]
    single = df.local_3.to_numpy() == mode
    out = {}

    # A. separability within 1-34
    tr = t <= 34
    for name, g in [("single", single), ("multi", ~single), ("all", np.ones(len(y), bool))]:
        m = tr & g
        scores = []
        for seed in (0, 1, 2):
            p = np.zeros(m.sum())
            Xm, ym = X[m], y[m]
            for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xm, ym):
                p[b] = model(seed).fit(Xm[a], ym[a]).predict_proba(Xm[b])[:, 1]
            scores.append(ap(ym, p))
        out[f"A_cv_pr_auc_1_34_{name}"] = {"mean": float(np.mean(scores)), "n": int(m.sum()),
                                           "illicit": int(y[m].sum()), "prevalence": float(y[m].mean())}

    # B. transfer to later multi-input rows
    rows = []
    for train_end in (34, 42):
        trm = t <= train_end
        for variant in ("global", "specialist", "upweight", "balanced"):
            for seed in (0, 1, 2):
                if variant == "specialist":
                    idx = np.flatnonzero(trm & ~single)
                    w = None
                else:
                    idx = np.flatnonzero(trm)
                    w = np.ones(len(idx))
                    if variant == "upweight":
                        w[(~single[idx]) & (y[idx] == 1)] = 10.0
                    if variant == "balanced":
                        for s in (True, False):
                            for c in (0, 1):
                                cell = (single[idx] == s) & (y[idx] == c)
                                w[cell] = len(idx) / (4 * max(1, cell.sum()))
                m = model(seed).fit(X[idx], y[idx], sample_weight=w)
                for per, pm in [("35-42", (t >= 35) & (t <= 42)), ("43-49", t >= 43)]:
                    if train_end == 42 and per == "35-42":
                        continue
                    for grp, gm in [("multi", ~single), ("all", np.ones(len(y), bool))]:
                        e = pm & gm
                        rows.append({"train": f"1-{train_end}", "variant": variant, "seed": seed, "eval": per,
                                     "group": grp, "n": int(e.sum()), "illicit": int(y[e].sum()),
                                     "pr_auc": ap(y[e], m.predict_proba(X[e])[:, 1])})
    b = pd.DataFrame(rows)
    b.to_csv(R / "minority_analysis_transfer.csv", index=False)
    bs = b.groupby(["train", "variant", "eval", "group"]).agg(pr_auc=("pr_auc", "mean"), n=("n", "first"),
                                                           illicit=("illicit", "first")).reset_index()

    # C. purity of the post-43 illicit clusters in the training period
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000, random_state=0).fit(X[t <= 34])
    pca = PCA(20, random_state=0).fit(qt.transform(X[t <= 34]))
    Z = pca.transform(qt.transform(X))
    ill = y == 1
    km = KMeans(12, n_init=10, random_state=0).fit(Z[ill])  # same as shift_analysis.py
    lab_all = km.predict(Z)
    post_share = pd.Series(lab_all[ill & (t >= 43)]).value_counts(normalize=True)
    top = list(post_share.index[:3])
    in_top = np.isin(lab_all, top)
    out["C_post_clusters"] = [int(c) for c in top]
    out["C_post_share_in_top3"] = float(post_share.iloc[:3].sum())
    for per, pm in [("1-34", t <= 34), ("35-42", (t >= 35) & (t <= 42)), ("43-49", t >= 43)]:
        sel = pm & in_top
        out[f"C_rows_in_top3_{per}"] = {"n": int(sel.sum()), "illicit": int(y[sel].sum()),
                                        "illicit_share": float(y[sel].mean()) if sel.any() else None}

    (R / "minority_analysis_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    bs.to_csv(R / "minority_analysis_transfer_summary.csv", index=False)
    print(json.dumps(out, indent=2))
    print(bs.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
