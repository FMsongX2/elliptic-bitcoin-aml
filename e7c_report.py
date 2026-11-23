"""Analyse E7c runs: selection scores, test results and how well each selection rule transfers.
Ensembles are evaluated in eval_reliability.py (rules that never look at a later time step).

Selection rules compared
  rolling  mean validation PR-AUC over folds cv1-cv3 (fit 1-20/1-26/1-30, validate 21-24/27-30/31-34)
  single   validation PR-AUC on cv3 only (31-34), the rule used by E6b/E7
Both are averaged over label shares 1/5/10% and selection seeds.

Outputs: results/e7c_summary.csv, results/e7c_selection.json, figures/e7c_*.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

from common import BASE

R = BASE / "results"
F = BASE / "figures"


def short(cid):
    c = json.loads(cid)
    if c["kind"] == "ls":
        return "LabelSpreading"
    s = f"{c['kind']}:{c['base']}"
    s += f",hl{c['half_life']}" if c["half_life"] else ""
    s += ",unsup" if c["unsup"] else ""
    if c["kind"] == "st":
        s += f",thr{c.get('thr_pos', c['thr'])}/{c['thr']}" + (",bal" if c["balanced"] else "") + (f",w{c['plw']}" if c["plw"] != 1 else "")
    return s


def table(runs):
    sel = runs[runs.split != "test"]
    test = runs[runs.split == "test"]
    rolling = sel.groupby("config").pr_auc.mean().rename("sel_rolling")
    single = sel[sel.split == "cv3"].groupby("config").pr_auc.mean().rename("sel_single")
    worst = sel.groupby(["config", "split"]).pr_auc.mean().groupby("config").min().rename("sel_worst_fold")
    t = test.groupby(["config", "frac"]).pr_auc.mean().unstack().add_prefix("test_")
    t["test_mean"] = test.groupby("config").pr_auc.mean()
    t["test_43_49"] = test.groupby("config").pr_auc_43_49.mean()
    out = pd.concat([rolling, single, worst, t], axis=1).reset_index()
    out.insert(1, "name", out.config.map(short))
    return out.sort_values("sel_rolling", ascending=False)


def main():
    runs = pd.read_csv(R / "e7c_runs.csv")
    tab = table(runs)
    tab.to_csv(R / "e7c_summary.csv", index=False, encoding="utf-8-sig")

    full = tab.dropna(subset=["test_mean"])
    rho_roll = spearmanr(full.sel_rolling, full.test_mean).correlation
    rho_single = spearmanr(full.sel_single, full.test_mean).correlation
    pick_roll = full.loc[full.sel_rolling.idxmax()]
    pick_single = full.loc[full.sel_single.idxmax()]
    info = {"n_configs": int(len(full)),
            "spearman_rolling_vs_test": float(rho_roll), "spearman_single_vs_test": float(rho_single),
            "pick_rolling": {"name": pick_roll["name"], "test_mean": float(pick_roll.test_mean)},
            "pick_single": {"name": pick_single["name"], "test_mean": float(pick_single.test_mean)},
            "best_on_test_for_reference": {"name": full.loc[full.test_mean.idxmax(), "name"],
                                           "test_mean": float(full.test_mean.max())}}
    (R / "e7c_selection.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(info, indent=2, ensure_ascii=False))
    pd.set_option("display.width", 250)
    print(tab.drop(columns="config").head(15).round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, col, rho, title in [(axes[0], "sel_single", rho_single, "Single validation 31-34 (E6b/E7 rule)"),
                                (axes[1], "sel_rolling", rho_roll, "Rolling-origin, 3 folds (E7c rule)")]:
        ax.scatter(full[col], full.test_mean, s=18, alpha=0.7)
        ax.set(title=f"{title}\nSpearman rho = {rho:.2f}", xlabel="Selection score (validation PR-AUC)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Test PR-AUC (35-49, mean of 1/5/10% labels)")
    fig.suptitle("E7c: does the validation score predict the test score? (one point = one configuration)")
    fig.tight_layout(); fig.savefig(F / "e7c_selection.png", dpi=130); plt.close(fig)
    return tab, info


if __name__ == "__main__":
    main()
