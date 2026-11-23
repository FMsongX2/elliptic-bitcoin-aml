"""E7c confirmation on fresh seeds.

The three E7c methods compared in the final evaluation (self-training, LightGBM, Label Spreading)
were picked after looking at stage 1-2 test results. To check how much of their result is
selection luck, re-run them on seeds 3, 4, 5, which no earlier experiment used (label selection
and model seeds both change). The stored predictions feed eval_reliability.py, which also builds
the ensembles with rules that never look at a later time step.

Splits: cv3 (fit 1-30 -> 31-34) and test (fit 1-34 -> 35-49). Label shares 1/5/10%.
Outputs: results/e7c_confirm_runs.csv, e7c_confirm_summary.csv, e7c_confirm_preds.joblib (not in git)
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from common import BASE, load_dataset
import e7c_search as e7c

R = BASE / "results"
SEEDS = [3, 4, 5]
FRACS = [0.01, 0.05, 0.10]
MEMBERS = {
    "ST": dict(kind="st", base="team", half_life=None, unsup=False, thr=0.95, balanced=False, plw=1.0),
    "LGBM": dict(kind="sup", base="team", half_life=None, unsup=False),
    "LS": dict(kind="ls"),
}


def main():
    D = e7c.Data(load_dataset(labeled_only=False))
    splits = [s for s in e7c.FOLDS if s[0] == "cv3"] + [e7c.TEST]
    preds, rows = {}, []
    for split in splits:
        for frac in FRACS:
            for seed in SEEDS:
                for name, cfg in MEMBERS.items():
                    p, ev, n_lab, n_ill = e7c.run(cfg, D, split, frac, seed)
                    preds[(name, split[0], frac, seed)] = (p, ev)
                print(f"{split[0]} frac={frac} seed={seed} done", flush=True)
    for (name, split_name, frac, seed), (p, ev) in preds.items():
        y = D.y[ev].astype(int)
        t = D.t[ev]
        row = {"method": name, "split": split_name, "frac": frac, "seed": seed,
               "pr_auc": average_precision_score(y, p)}
        if split_name == "test":
            row["pr_auc_35_42"] = average_precision_score(y[t < 43], p[t < 43])
            row["pr_auc_43_49"] = average_precision_score(y[t >= 43], p[t >= 43])
        rows.append(row)
    runs = pd.DataFrame(rows)
    runs.to_csv(R / "e7c_confirm_runs.csv", index=False)
    joblib.dump({k: v[0].astype(np.float32) for k, v in preds.items()}, R / "e7c_confirm_preds.joblib")
    summ = runs.groupby(["method", "split", "frac"]).pr_auc.agg(["mean", "std"]).unstack(["split", "frac"])
    summ.to_csv(R / "e7c_confirm_summary.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(summ.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
