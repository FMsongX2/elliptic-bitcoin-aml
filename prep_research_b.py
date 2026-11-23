"""Preprocessing research for setting B (57 transaction-time features), for distance-based
semi-supervised methods (kNN, Label Spreading, K-Means cluster representatives).

Uses ONLY train (steps 1-30, all rows incl. unknown, for fitting) and validation (31-34, labeled)
rows. The test steps 35-49 are never loaded into any metric.

Candidate pipelines (all followed by PCA-20 for the distance space, as in E7):
  std_57            StandardScaler on 57
  quantile_57       QuantileTransformer(normal) on 57 (the current E7 embedding)
  quantile_54       same without time proxy agg_6/7/9
  log_54            log1p(x - train_min), then StandardScaler
  yeojohnson_54     PowerTransformer(yeo-johnson), standardized
  split_54          features with a dominant value (share > 0.5 on train) are split into
                    a 0/1 "is dominant value" indicator (identical indicators de-duplicated)
                    and the quantile-normal value of the remaining rows (dominant rows -> 0)
  split_dedup_54    split_54, then drop near-duplicate columns (|r| > 0.95, greedy, train)

Metrics per pipeline:
  label-free: median |skew|, median excess kurtosis, share of near-Gaussian columns
              (|skew| < 0.5 and |kurt| < 1) on the transformed train rows (before PCA)
  val PR-AUC: kNN (k=15, distance-weighted) trained on 1% / 10% / 100% of train labels
              Label Spreading at 1% / 10% (E7 code), cluster representatives at 5% (E7 code)
Seeds 0, 1, 2. Output: results/prep_research_B.csv, figures/prep_research_B.png
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler

from common import BASE
import common
import e7_semi_supervised as e7

TIME_PROXY = ["agg_6", "agg_7", "agg_9"]
SEEDS = [0, 1, 2]
PIPELINES = ["std_57", "quantile_57", "quantile_54", "log_54", "yeojohnson_54", "split_54", "split_dedup_54"]


# ---------------------------------------------------------------- transformers (fit on train only)
class LogShift:
    def fit(self, X):
        self.min_ = X.min(axis=0)
        self.sc = StandardScaler().fit(np.log1p(np.maximum(X - self.min_, 0)))
        return self

    def transform(self, X):
        return self.sc.transform(np.log1p(np.maximum(X - self.min_, 0)))


class SplitDominant:
    """Dominant-value indicator + quantile-normal value of the other rows."""

    def __init__(self, share=0.5, seed=0):
        self.share, self.seed = share, seed

    def fit(self, X):
        n, d = X.shape
        self.mode_ = np.zeros(d)
        self.split_ = np.zeros(d, bool)
        for j in range(d):
            vals, cnt = np.unique(X[:, j], return_counts=True)
            k = cnt.argmax()
            self.mode_[j] = vals[k]
            self.split_[j] = cnt[k] / n > self.share
        ind = (X[:, self.split_] == self.mode_[self.split_]).astype(np.float32)
        # keep one indicator per identical pattern (e.g. "single input" flags shared by many features)
        _, first = np.unique(ind.T, axis=0, return_index=True)
        self.ind_keep_ = np.sort(first)
        self.qts_ = []
        for j in range(d):
            col = X[:, j]
            rest = col[col != self.mode_[j]] if self.split_[j] else col
            if len(rest) < 10:
                rest = col
            self.qts_.append(QuantileTransformer(output_distribution="normal", n_quantiles=min(1000, len(rest)),
                                                 subsample=100000, random_state=self.seed).fit(rest.reshape(-1, 1)))
        self.sc = StandardScaler().fit(self._raw(X))
        return self

    def _raw(self, X):
        vals = np.empty_like(X, dtype=np.float64)
        for j, qt in enumerate(self.qts_):
            v = qt.transform(X[:, j:j + 1]).ravel()
            if self.split_[j]:
                v[X[:, j] == self.mode_[j]] = 0.0
            vals[:, j] = v
        ind = (X[:, self.split_] == self.mode_[self.split_]).astype(np.float64)[:, self.ind_keep_]
        return np.hstack([vals, ind])

    def transform(self, X):
        return self.sc.transform(self._raw(X))


class DropCollinear:
    def __init__(self, base, thr=0.95, seed=0):
        self.base, self.thr, self.seed = base, thr, seed

    def fit(self, X):
        self.base.fit(X)
        Z = self.base.transform(X)
        rng = np.random.RandomState(self.seed)
        idx = rng.choice(len(Z), min(30000, len(Z)), replace=False)
        with np.errstate(invalid="ignore"):
            C = np.abs(np.corrcoef(Z[idx].T))
        C = np.nan_to_num(C)
        keep = []
        for j in range(Z.shape[1]):
            if all(C[j, k] <= self.thr for k in keep):
                keep.append(j)
        self.keep_ = np.array(keep)
        return self

    def transform(self, X):
        return self.base.transform(X)[:, self.keep_]


class Pipe:
    """Column selection (57 or 54) + transformer + PCA-20. transform() takes raw 57-column X."""

    def __init__(self, name, cols57, seed=0):
        self.name, self.seed = name, seed
        self.sel = np.array([c not in TIME_PROXY for c in cols57]) if name.endswith("_54") else np.ones(len(cols57), bool)
        kind = name.rsplit("_", 1)[0]
        qt = lambda: QuantileTransformer(output_distribution="normal", n_quantiles=1000, subsample=100000, random_state=seed)
        self.t = {"std": StandardScaler(), "quantile": qt(), "log": LogShift(),
                  "yeojohnson": PowerTransformer(method="yeo-johnson", standardize=True),
                  "split": SplitDominant(seed=seed), "split_dedup": DropCollinear(SplitDominant(seed=seed), seed=seed)}[kind]

    def fit(self, X):
        Xs = X[:, self.sel].astype(np.float64)
        if isinstance(self.t, PowerTransformer):  # Yeo-Johnson lambda search is slow on 120k rows
            rng = np.random.RandomState(self.seed)
            self.t.fit(Xs[rng.choice(len(Xs), min(50000, len(Xs)), replace=False)])
        else:
            self.t.fit(Xs)
        T = np.nan_to_num(self.t.transform(Xs), posinf=0, neginf=0)
        self.pca = PCA(n_components=20, random_state=self.seed).fit(T)
        self.T_train_ = T
        return self

    def features(self, X):
        return np.nan_to_num(self.t.transform(X[:, self.sel].astype(np.float64)), posinf=0, neginf=0)

    def transform(self, X):
        return self.pca.transform(self.features(X)).astype(np.float32)


# ---------------------------------------------------------------- driver
def gaussianity(T, seed=0):
    rng = np.random.RandomState(seed)
    S = T[rng.choice(len(T), min(50000, len(T)), replace=False)]
    sk, ku = stats.skew(S), stats.kurtosis(S)
    ok = (np.abs(sk) < 0.5) & (np.abs(ku) < 1)
    return float(np.nanmedian(np.abs(sk))), float(np.nanmedian(ku)), float(np.mean(ok)), int(T.shape[1])


def main(pipelines=PIPELINES, seeds=SEEDS, ls_fracs=(0.01, 0.10), knn_fracs=(0.01, 0.10, 1.0), rep_frac=0.05):
    df = common.load_dataset(labeled_only=False)
    cols = common.feature_cols(df, "time")
    tr = df[df["time_step"].isin(list(common.TRAIN_STEPS))]
    va = df[df["time_step"].isin(list(common.VAL_STEPS)) & df["y"].notna()]
    X_tr = tr[cols].to_numpy(np.float32)
    y_tr = tr["y"].to_numpy()
    pool = np.flatnonzero(pd.notna(y_tr))
    pool_y = y_tr[pool].astype(int)
    unk = np.flatnonzero(pd.isna(y_tr))
    X_va, y_va = va[cols].to_numpy(np.float32), va["y"].to_numpy(int)
    e7.UNLABELED_CAP = 30000
    print(f"train rows {len(X_tr)} (labeled {len(pool)}), val labeled {len(y_va)}", flush=True)

    rows = []
    for name in pipelines:
        t0 = time.time()
        p = Pipe(name, cols, seed=0).fit(X_tr)
        msk, mku, gshare, ncols = gaussianity(p.T_train_)
        Z_tr, Z_va = p.transform(X_tr), p.transform(X_va)
        base = dict(pipeline=name, n_columns=ncols, median_abs_skew=msk, median_kurtosis=mku, gaussian_share=gshare)
        print(f"[{name}] cols={ncols} |skew|={msk:.2f} kurt={mku:.2f} gaussian={gshare:.2f} ({time.time()-t0:.0f}s)", flush=True)
        for seed in seeds:
            rng = np.random.RandomState(seed)
            r = dict(base, seed=seed)
            for f in knn_fracs:
                q = rng.choice(len(pool), max(20, int(round(f * len(pool)))), replace=False)
                if len(np.unique(pool_y[q])) < 2:
                    r[f"knn_{f:g}"] = np.nan
                    continue
                knn = KNeighborsClassifier(n_neighbors=15, weights="distance").fit(Z_tr[pool[q]], pool_y[q])
                r[f"knn_{f:g}"] = average_precision_score(y_va, knn.predict_proba(Z_va)[:, 1])
            for method, fracs in [("label_spreading", ls_fracs), ("cluster_rep", (rep_frac,))]:
                for f in fracs:
                    budget = max(20, int(round(f * len(pool))))
                    m, _, _ = e7.train_method(method, X_tr, pool, pool_y, unk, budget, seed, {"emb": p, "Z": Z_tr})
                    r[f"{method}_{f:g}"] = average_precision_score(y_va, m.predict_proba(X_va)[:, 1])
            rows.append(r)
            print(f"  seed {seed}: " + " ".join(f"{k}={v:.3f}" for k, v in r.items() if k.startswith(("knn", "label", "cluster"))) +
                  f" ({time.time()-t0:.0f}s)", flush=True)

    res = pd.DataFrame(rows)
    out = BASE / "results" / "prep_research_B.csv"
    res.to_csv(out, index=False, encoding="utf-8-sig")
    metric_cols = [c for c in res.columns if c.startswith(("knn", "label", "cluster"))]
    summ = res.groupby("pipeline", sort=False)[["n_columns", "median_abs_skew", "median_kurtosis", "gaussian_share"] + metric_cols].mean()
    summ.join(res.groupby("pipeline", sort=False)[metric_cols].std().add_suffix("_std")).to_csv(
        BASE / "results" / "prep_research_B_summary.csv", encoding="utf-8-sig")
    print(summ.round(3).to_string(), flush=True)

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(summ))
    w = 0.8 / len(metric_cols)
    for i, c in enumerate(metric_cols):
        ax.bar(x + i * w, summ[c], w, yerr=res.groupby("pipeline", sort=False)[c].std(), capsize=2, label=c)
    ax.axhline(y_va.mean(), color="gray", ls=":", label=f"Random guess ({y_va.mean():.3f})")
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels(summ.index, rotation=15)
    ax.set(ylabel="Validation PR-AUC (steps 31-34)", title="Preprocessing for distance-based semi-supervised methods (setting B, seeds 0-2)")
    ax.legend(fontsize=7.5, ncol=4, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(BASE / "figures" / "prep_research_B.png", dpi=130)
    plt.close(fig)
    return res


if __name__ == "__main__":
    main()
