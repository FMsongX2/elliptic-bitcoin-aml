"""E9 hedge: base model + multi-input specialist, combined without using any later time step
(past = ranks within steps up to the scored one, prob = mean probability).
Output: results/e9_hedge.csv"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from common import BASE
import common
import e9_minority_fix as e9

df = common.load_dataset(labeled_only=True)
cols = [c for c in common.feature_cols(df, "time") if c not in e9.TIME_PROXY]
X, t, y = df[cols].to_numpy(np.float32), df.time_step.to_numpy(), df.y.to_numpy(int)
single = df.local_3.to_numpy() == df.loc[t <= 30, "local_3"].value_counts().index[0]
rows = []
for end in (34, 42):
    for seed in (0, 1, 2):
        tr = t <= end
        g, sp = e9.lgbm(seed).fit(X[tr], y[tr]), e9.lgbm(seed).fit(X[tr & ~single], y[tr & ~single])
        te = t > end
        te &= t >= (43 if end == 42 else 35)
        tt, yy = t[te], y[te]
        pg, ps = g.predict_proba(X[te])[:, 1], sp.predict_proba(X[te])[:, 1]
        past = np.zeros(len(yy))
        for s in np.unique(tt):
            ref = tt <= s
            now = tt[ref] == s
            past[tt == s] = ((rankdata(pg[ref]) + rankdata(ps[ref])) / (2 * ref.sum()))[now]
        for per, m in (("35-42", tt < 43), ("43-49", tt >= 43), ("35-49", tt >= 35)):
            if not m.any() or (end == 42 and per != "43-49"):
                continue
            v = dict(global_model=pg[m], specialist=ps[m], hedge_past=past[m], hedge_prob=(pg[m] + ps[m]) / 2)
            rows += [dict(train=f"1-{end}", eval=per, method=k, ap=e9.ap(yy[m], p)) for k, p in v.items()]
r = pd.DataFrame(rows).groupby(["train", "eval", "method"]).ap.mean().unstack("method")
r.to_csv(BASE / "results" / "e9_hedge.csv")
print(r.round(3).to_string())
