"""E8: recovery after the distribution shift at step 43 (dark market shutdown), setting B.

Every method in E6b-B / E7 collapses on steps 43-49 (PR-AUC 0.02-0.11). E8 asks how much
post-shift labelled data is needed to recover, and whether old data helps or hurts.

Part A (fixed test = labelled steps 46-49; 116 illicit / 2,505):
  train_1_34           all labels of 1-34 (the E2/E7 protocol)
  train_1_42           all labels of 1-42 (catch up to just before the shift)
  train_1_42_post{f}   1-42 + a random share f of the labelled rows of 43-45, f in 0.1/0.25/0.5/1.0,
                       post-shift rows weighted w = 1 or 10
  train_43_45          only the labelled rows of 43-45
Part B (rolling, one step ahead): for every test step t in 35-49 train on labels of steps < t
  static     1-34 only (no retraining)
  expanding  1..t-1
  window10   t-10..t-1 (forget old patterns)

Model: LightGBM with the team E2 parameters (e7.make_lgbm). Seeds 0, 1, 2.
Metric: PR-AUC only. After the shift there is no separate validation period, so no threshold
can be tuned fairly and F1 is not reported.
Outputs: results/e8_partA.csv, e8_partB.csv, e8_summary.csv, results.csv rows "E8",
figures/e8_recovery.png, e8_rolling.png.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from common import BASE
import common
from common import clear_results, feature_cols, load_dataset, log_result
import e7_semi_supervised as e7

RESULTS_DIR = BASE / "results"
FIG_DIR = BASE / "figures"
SEEDS = [0, 1, 2, 3, 4, 5]
POST_FRACS = [0.1, 0.25, 0.5, 1.0]
WEIGHTS = [1, 10]


def fit(X, y, w, seed):
    if len(np.unique(y)) < 2:
        return e7.PriorModel(np.mean(y))
    return e7.make_lgbm(seed).fit(X, y, sample_weight=w)


def pr_auc(y, p):
    return float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def part_a(df, cols, seeds):
    t = df["time_step"].to_numpy()
    X, y = df[cols].to_numpy(np.float32), df["y"].to_numpy(int)
    test = (t >= 46)
    pre34, pre42, post = (t <= 34), (t <= 42), (t >= 43) & (t <= 45)
    rows = []
    for seed in seeds:
        rng = np.random.RandomState(seed)
        post_idx = np.flatnonzero(post)
        conds = [("train_1_34", np.flatnonzero(pre34), None), ("train_1_42", np.flatnonzero(pre42), None),
                 ("train_43_45", post_idx, None)]
        for f in POST_FRACS:
            pick = rng.choice(post_idx, int(round(f * len(post_idx))), replace=False)
            for w in WEIGHTS:
                idx = np.concatenate([np.flatnonzero(pre42), pick])
                wt = np.concatenate([np.ones(pre42.sum()), np.full(len(pick), float(w))])
                conds.append((f"train_1_42_post{f:g}_w{w}", idx, wt))
        for name, idx, wt in conds:
            m = fit(X[idx], y[idx], wt, seed)
            p = m.predict_proba(X[test])[:, 1]
            post_rows = np.isin(idx, post_idx)
            rows.append({"condition": name, "seed": seed, "n_train": len(idx),
                         "n_post_labels": int(post_rows.sum()), "n_post_illicit": int(y[idx][post_rows].sum()),
                         "pr_auc_46_49": pr_auc(y[test], p),
                         **{f"pr_auc_{s}": pr_auc(y[t == s], p[t[test] == s]) for s in range(46, 50)}})
            print(f"[A] {name} seed={seed} post_labels={rows[-1]['n_post_labels']} "
                  f"pr_auc_46_49={rows[-1]['pr_auc_46_49']:.3f}", flush=True)
    return pd.DataFrame(rows)


def part_b(df, cols, seeds):
    t = df["time_step"].to_numpy()
    X, y = df[cols].to_numpy(np.float32), df["y"].to_numpy(int)
    rows = []
    for seed in seeds:
        static = fit(X[t <= 34], y[t <= 34], None, seed)
        for s in range(35, 50):
            te = t == s
            for name, tr in [("static", t <= 34), ("expanding", t < s), ("window10", (t < s) & (t >= s - 10))]:
                m = static if name == "static" else fit(X[tr], y[tr], None, seed)
                rows.append({"variant": name, "seed": seed, "test_step": s, "n_train": int(tr.sum()),
                             "n_illicit_test": int(y[te].sum()), "pr_auc": pr_auc(y[te], m.predict_proba(X[te])[:, 1])})
        print(f"[B] seed={seed} done", flush=True)
    return pd.DataFrame(rows)


def log_part_a(a):
    """results.csv rows for part A. log_result appends by position, so every metric column must be given."""
    clear_results("E8")
    nan = float("nan")
    for r in a.itertuples():
        log_result("E8", model=f"LightGBM {r.condition}", feature_set="time", seed=int(r.seed),
                   metrics={"precision": nan, "recall": nan, "f1": nan, "pr_auc": r.pr_auc_46_49, "threshold": nan},
                   note=f"partA;test=46-49;n_post_labels={r.n_post_labels};n_post_illicit={r.n_post_illicit}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    args = ap.parse_args(argv)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset(labeled_only=True)
    cols = feature_cols(df, "time")
    a = part_a(df, cols, args.seeds)
    b = part_b(df, cols, args.seeds)
    a.to_csv(RESULTS_DIR / "e8_partA.csv", index=False, encoding="utf-8-sig")
    b.to_csv(RESULTS_DIR / "e8_partB.csv", index=False, encoding="utf-8-sig")

    sa = a.groupby("condition", sort=False).agg(n_post_labels=("n_post_labels", "mean"),
                                                 n_post_illicit=("n_post_illicit", "mean"),
                                                 pr_auc_mean=("pr_auc_46_49", "mean"), pr_auc_std=("pr_auc_46_49", "std"))
    sb = b.assign(period=np.where(b.test_step >= 43, "43-49", "35-42")).groupby(["variant", "period"]) \
          .pr_auc.agg(["mean", "std"]).unstack()
    with open(RESULTS_DIR / "e8_summary.csv", "w", encoding="utf-8-sig", newline="") as f:
        f.write("# Part A: test steps 46-49\n")
        sa.to_csv(f)
        f.write("# Part B: one-step-ahead PR-AUC averaged over steps (seed mean)\n")
        sb.to_csv(f)
    print(sa.round(3).to_string())
    print(sb.round(3).to_string())

    log_part_a(a)

    # Figure A
    fig, ax = plt.subplots(figsize=(9, 4.8))
    base = sa.loc[["train_1_34", "train_1_42", "train_43_45"]]
    for i, (name, r) in enumerate(base.iterrows()):
        ax.axhline(r.pr_auc_mean, ls=["--", "-.", ":"][i], color=["gray", "black", "tab:purple"][i],
                   label=f"{name} ({r.pr_auc_mean:.3f})")
    for w, color in zip(WEIGHTS, ["tab:blue", "tab:red"]):
        d = sa.loc[[f"train_1_42_post{f:g}_w{w}" for f in POST_FRACS]]
        ax.errorbar(d.n_post_labels, d.pr_auc_mean, yerr=d.pr_auc_std, marker="o", capsize=3, color=color,
                    label=f"1-42 + share of 43-45 labels, weight {w}")
        for (_, r), f in zip(d.iterrows(), POST_FRACS):
            ax.annotate(f"{f:.0%}\n({r.n_post_illicit:.0f} illicit)", (r.n_post_labels, r.pr_auc_mean),
                        xytext=(4, 6), textcoords="offset points", fontsize=7, color=color)
    prev = df[df.time_step >= 46].y.mean()
    ax.axhline(prev, color="lightgray", ls=":", label=f"Random guess ({prev:.3f})")
    ax.set(xlabel="Labelled rows from steps 43-45 added to training", ylabel="PR-AUC on steps 46-49",
           title="E8-A: post-shutdown labels needed to recover (setting B, LightGBM, seeds 0-2)")
    ax.grid(alpha=0.3); ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG_DIR / "e8_recovery.png", dpi=130); plt.close(fig)

    # Figure B
    fig, ax1 = plt.subplots(figsize=(10, 4.8)); ax2 = ax1.twinx()
    cnt = b[b.seed == b.seed.min()].drop_duplicates("test_step").set_index("test_step").n_illicit_test
    ax2.bar(cnt.index, cnt.values, color="lightgray", alpha=0.5); ax2.set_ylabel("Illicit count")
    for v in ["static", "expanding", "window10"]:
        d = b[b.variant == v].groupby("test_step").pr_auc.mean()
        ax1.plot(d.index, d.values, marker="o", label=v)
    ax1.axvline(43, color="black", ls="--", label="Dark market shutdown (43)")
    ax1.set(xlabel="Test step (trained on labels of earlier steps)", ylabel="PR-AUC",
            title="E8-B: retraining every step, one step ahead (setting B, LightGBM, seeds 0-2)")
    ax1.set_zorder(ax2.get_zorder() + 1); ax1.patch.set_visible(False)
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG_DIR / "e8_rolling.png", dpi=130); plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
