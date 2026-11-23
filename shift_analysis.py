"""Did the illicit pattern change at step 43, or was the post-43 pattern already present
before as a small minority? (setting B features without the time proxies agg_6/7/9)

1. Domain classifier: how separable are illicit 1-42 vs illicit 43-49, compared with licit
   1-42 vs licit 43-49 (5-fold CV ROC-AUC, LightGBM). Much higher for illicit -> illicit changed
   beyond the general drift.
2. Neighbours: for labelled transactions of 35-42 and 43-49, share of illicit among the 10
   nearest labelled neighbours in steps 1-34 (quantile + PCA-20 space fitted on 1-34).
3. Clusters: K-Means (k=12) on all illicit transactions; share of each period per cluster.
   A post-43 cluster with a small pre-43 share = minority pattern; with ~0 = new pattern.
4. Features: KS statistic pre vs post, for illicit and for licit.
Outputs: results/shift_analysis_*.csv, figures/shift_analysis.png
"""
import json

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer

from common import BASE
from common import feature_cols, load_dataset

R = BASE / "results"
TIME_PROXY = ["agg_6", "agg_7", "agg_9"]


def main():
    df = load_dataset(labeled_only=True)
    cols = [c for c in feature_cols(df, "time") if c not in TIME_PROXY]
    t, y = df.time_step.to_numpy(), df.y.to_numpy()
    X = df[cols].to_numpy(np.float32)
    pre, post = t <= 42, t >= 43
    out = {}

    # 1. domain classifiers
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    for name, cls in [("illicit", 1), ("licit", 0)]:
        m = y == cls
        Xd, yd = X[m], post[m].astype(int)
        if name == "licit":  # subsample licit to the illicit size so both tasks are comparable
            rng = np.random.RandomState(0)
            n_post = int(((y == 1) & post).sum())
            n_pre = int(((y == 1) & pre).sum())
            idx = np.concatenate([rng.choice(np.flatnonzero(yd == 0), n_pre, replace=False),
                                  rng.choice(np.flatnonzero(yd == 1), n_post, replace=False)])
            Xd, yd = Xd[idx], yd[idx]
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, verbose=-1, random_state=0)
        out[f"domain_auc_{name}"] = float(cross_val_score(clf, Xd, yd, cv=cv, scoring="roc_auc").mean())
        out[f"n_{name}_pre_post"] = [int((yd == 0).sum()), int((yd == 1).sum())]

    # 2. neighbours in the training period
    tr = t <= 34
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000, random_state=0).fit(X[tr])
    pca = PCA(20, random_state=0).fit(qt.transform(X[tr]))
    Z = pca.transform(qt.transform(X))
    nn = NearestNeighbors(n_neighbors=10).fit(Z[tr])
    rows = []
    for per, pm in [("35-42", (t >= 35) & (t <= 42)), ("43-49", post)]:
        for cls, lab in [(1, "illicit"), (0, "licit")]:
            q = pm & (y == cls)
            _, idx = nn.kneighbors(Z[q])
            share = y[tr][idx].mean(axis=1)
            rows.append({"period": per, "query": lab, "n": int(q.sum()),
                         "mean_illicit_share_of_neighbours": float(share.mean()),
                         "share_with_majority_illicit_neighbours": float((share >= 0.5).mean())})
    nb = pd.DataFrame(rows)
    nb.to_csv(R / "shift_analysis_neighbours.csv", index=False)

    # 3. clusters of illicit transactions
    ill = y == 1
    km = KMeans(12, n_init=10, random_state=0).fit(Z[ill])
    lab = km.labels_
    per = np.where(t[ill] <= 34, "1-34", np.where(t[ill] <= 42, "35-42", "43-49"))
    ct = pd.crosstab(lab, per)
    share = ct / ct.sum(axis=0)
    share.columns = [f"share_of_{c}" for c in share.columns]
    cl = ct.join(share).sort_values("share_of_43-49", ascending=False)
    cl.index.name = "cluster"
    cl.to_csv(R / "shift_analysis_clusters.csv")

    # 4. features
    ks = []
    for c_i, c in enumerate(cols):
        ks.append({"feature": c,
                   "ks_illicit_pre_post": ks_2samp(X[ill & pre, c_i], X[ill & post, c_i]).statistic,
                   "ks_licit_pre_post": ks_2samp(X[(y == 0) & pre, c_i], X[(y == 0) & post, c_i]).statistic})
    ks = pd.DataFrame(ks).sort_values("ks_illicit_pre_post", ascending=False)
    ks.to_csv(R / "shift_analysis_features.csv", index=False)

    (R / "shift_analysis_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(nb.round(3).to_string(index=False))
    print(cl.round(3).to_string())
    print(ks.head(10).round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    s = cl[[c for c in cl.columns if c.startswith("share_of")]]
    s.plot(kind="bar", ax=axes[0])
    axes[0].set(title="Illicit clusters: share of each period's illicit transactions", xlabel="Cluster (sorted by 43-49 share)",
                ylabel="Share")
    top = ks.head(12)
    xs = np.arange(len(top))
    axes[1].barh(xs - 0.2, top.ks_illicit_pre_post, 0.4, label="illicit 1-42 vs 43-49")
    axes[1].barh(xs + 0.2, top.ks_licit_pre_post, 0.4, label="licit 1-42 vs 43-49")
    axes[1].set_yticks(xs); axes[1].set_yticklabels(top.feature); axes[1].invert_yaxis()
    axes[1].set(title="Features that changed most for illicit (KS)", xlabel="KS statistic"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(BASE / "figures" / "shift_analysis.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
