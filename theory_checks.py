"""Does each textbook explanation of the results make a prediction that holds in this data?

Post-hoc: run after the results were known, on the same test period, so agreement supports an
explanation but is not an independent confirmation. Nothing is selected here.

1. learning curve    same-period hold-out (random 20% of the labelled rows of 1-34) vs later steps.
                     The random hold-out is optimistic (near-duplicate rows on both sides); it is
                     only a contrast for "too little data" vs "different period", never a result.
2. pseudo labels     how right are predictions made with probability >= 0.95 / <= 0.05
3. shift type        single-input group: observed P(illicit | group) vs the value implied if only
                     the class prior had moved (label shift)
4. ensemble          correlation of the members' errors (score - label), and member PR-AUC
5. base rate         ROC-AUC per period, which does not depend on the illicit share
6. anomaly premise   distance to the 10th nearest neighbour (standardised, no time proxies)
Outputs: results/theory_{learning_curve,pseudo_labels,shift,ensemble,roc_auc,distance}.csv
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import joblib

from common import BASE
from common import load_dataset
import e7c_search as e7c
import eval_reliability as er

R = BASE / "results"
SEEDS = range(6)
FRACS = [0.01, 0.05, 0.10, 0.50, 1.0]


def save(rows, name):
    out = pd.DataFrame(rows)
    out.to_csv(R / f"theory_{name}.csv", index=False, encoding="utf-8-sig")
    print(f"## {name}\n{out.round(3).to_string(index=False)}", flush=True)


def prior_only(p_old, prior_old, prior_new):
    """P(y=1 | group) after a pure class-prior change (Bayes rule on the odds)."""
    odds = p_old / (1 - p_old) * (prior_new / (1 - prior_new)) / (prior_old / (1 - prior_old))
    return odds / (1 + odds)


def main():
    df = load_dataset(labeled_only=False)
    D = e7c.Data(df)
    fit, pool, _, ev = D.split(range(1, 35), range(35, 50))
    y, yt, t = D.y[pool].astype(int), D.y[ev].astype(int), D.t[ev]
    periods = {"35-42": t < 43, "43-49": t >= 43}

    perm = np.random.RandomState(99).permutation(len(pool))
    hold, rest = perm[: len(pool) // 5], perm[len(pool) // 5:]
    runs, conf = [], []
    for f in FRACS:
        for s in SEEDS:
            q = np.random.RandomState(s).permutation(rest)[: min(len(rest), int(round(f * len(pool))))]
            m = e7c.lgbm("team", s).fit(D.X[pool[q]], y[q])
            ph, pf = m.predict_proba(D.X[pool[hold]])[:, 1], m.predict_proba(D.X[ev])[:, 1]
            sets = [("same period", ph, y[hold])] + [(k, pf[v], yt[v]) for k, v in periods.items()]
            for where, p, truth in sets:
                runs.append(dict(frac=f, n_labels=len(q), where=where, pr_auc=average_precision_score(truth, p),
                                 roc_auc=roc_auc_score(truth, p)))
                if f <= 0.10:
                    hi, lo = p >= 0.95, p <= 0.05
                    conf.append(dict(frac=f, where=where, n_confident_illicit=int(hi.sum()),
                                     precision_confident_illicit=truth[hi].mean() if hi.any() else np.nan,
                                     illicit_confidently_called_licit=(lo & (truth == 1)).sum() / truth.sum(),
                                     share_confident=(hi | lo).mean()))
    g = pd.DataFrame(runs).groupby(["frac", "n_labels", "where"])
    save(g.pr_auc.agg(["mean", "std"]).join(g.roc_auc.mean()).reset_index(), "learning_curve")
    save(pd.DataFrame(conf).groupby(["frac", "where"]).mean().reset_index(), "pseudo_labels")

    single = df.local_3.to_numpy() == df.loc[df.time_step <= 30, "local_3"].value_counts().index[0]
    prior0, rows = y.mean(), []
    for per, yy, g in [("1-34", y, single[pool])] + [(k, yt[v], single[ev][v]) for k, v in periods.items()]:
        for name, mask, mask0 in (("single", g, single[pool]), ("multi", ~g, ~single[pool])):
            rows.append(dict(period=per, group=name, prior=yy.mean(), p_illicit_in_group=yy[mask].mean(),
                             expected_if_prior_only=prior_only(y[mask0].mean(), prior0, yy.mean()),
                             share_of_illicit=mask[yy == 1].mean(), share_of_licit=mask[yy == 0].mean(),
                             roc_auc_of_group_flag=roc_auc_score(yy, mask)))
    save(rows, "shift")

    a, b = joblib.load(R / "e7c_preds.joblib"), joblib.load(R / "e7c_confirm_preds.joblib")
    get = lambda m, f, s: (a[(er.CID[m], "test", f, s)] if s < 3 else b[(m, "test", f, s)])
    ens, roc = [], []
    for f in e7c.FRACS:
        row = {"frac": f}
        for x, z in (("ST", "LGBM"), ("ST", "LS"), ("LGBM", "LS")):
            row[f"error_corr_{x}_{z}"] = np.mean([np.corrcoef(get(x, f, s) - yt, get(z, f, s) - yt)[0, 1] for s in SEEDS])
        for m in er.CID:
            row[f"pr_auc_{m}"] = np.mean([average_precision_score(yt, get(m, f, s)) for s in SEEDS])
            roc += [dict(method=m, frac=f, period=k,
                         roc_auc=np.mean([roc_auc_score(yt[v], get(m, f, s)[v]) for s in SEEDS]))
                    for k, v in periods.items()]
        ens.append(row)
    save(ens, "ensemble")
    save(roc, "roc_auc")

    sc = StandardScaler().fit(D.X[fit][:, D.notp])
    ref = np.sort(np.random.RandomState(0).choice(fit, min(30000, len(fit)), replace=False))
    nn = NearestNeighbors(n_neighbors=11).fit(sc.transform(D.X[ref][:, D.notp]))
    rows = []
    for per, idx, yy in [("1-34", pool, y)] + [(k, ev[v], yt[v]) for k, v in periods.items()]:
        d = nn.kneighbors(sc.transform(D.X[idx][:, D.notp]))[0]
        d10 = np.where(np.isin(idx, ref), d[:, 10], d[:, 9])  # skip the row itself when it is in the reference
        rows.append(dict(period=per, median_illicit=np.median(d10[yy == 1]), median_licit=np.median(d10[yy == 0]),
                         roc_auc_distance_to_illicit=roc_auc_score(yy, d10)))
    save(rows, "distance")


if __name__ == "__main__":
    main()
