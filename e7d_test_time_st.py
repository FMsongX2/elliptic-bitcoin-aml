"""E7d: causal test-time self-training (setting B).

E7c showed that configuration tweaks inside steps 1-34 move test PR-AUC by less than the noise,
and E8 showed that the step-43 drift is the real bottleneck. Labels after step 34 are not
available in the label-scarce setting, but UNLABELLED transactions are: when the transactions of
step t are scored, their features (and those of earlier steps) are already known.

For every test step t in 35-49:
  labels     the same revealed share (1/5/10%) of the labelled rows of steps 1-34 (seeded)
  unlabelled 'static'   unknown + unrevealed rows of steps 1-34 (E7 self-training)
             'causal'   the same plus ALL rows of steps 35..t (no labels), i.e. what is known
                        when step t is scored; nothing after t is used
  model      self-training with LightGBM (team E2 parameters), threshold 0.95, 5 rounds,
             unlabelled pool capped at 30,000 rows, of which the rows of steps 35..t are always
             kept (up to 15,000) so the new period is represented
Then PR-AUC on the labelled rows of step t, and pooled over 35-42 / 43-49 / 35-49.
Seeds 0, 1, 2. Outputs: results/e7d_runs.csv, e7d_summary.csv, figures/e7d_by_step.png
"""
import argparse
import csv
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from common import BASE
from common import feature_cols, load_dataset
import e7c_search as e7c

R = BASE / "results"
F = BASE / "figures"
FRACS = [0.01, 0.05, 0.10]
SEEDS = [0, 1, 2]
CAP = 30000
NEW_CAP = 15000


def self_train(XL, yL, XU, seed, thr=0.95, rounds=5):
    alive = np.ones(len(XU), bool)
    m = None
    for _ in range(rounds):
        m = e7c.lgbm("team", seed).fit(XL, yL)
        if not alive.any():
            break
        idx = np.flatnonzero(alive)
        p = m.predict_proba(XU[idx])[:, 1]
        sure = (p >= thr) | (p <= 1 - thr)
        add = idx[sure]
        if len(add) == 0:
            break
        XL = np.vstack([XL, XU[add]])
        yL = np.concatenate([yL, (p[sure] >= 0.5).astype(int)])
        alive[add] = False
    else:  # round limit reached: the last pseudo labels still need a fit (as in sklearn)
        m = e7c.lgbm("team", seed).fit(XL, yL)
    return m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fracs", nargs="*", type=float, default=FRACS)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--steps", nargs="*", type=int, default=list(range(35, 50)))
    args = ap.parse_args(argv)
    (R / "e7d_preds").mkdir(parents=True, exist_ok=True)
    df = load_dataset(labeled_only=False)
    X = df[feature_cols(df, "time")].to_numpy(np.float32)
    t = df["time_step"].to_numpy()
    y = df["y"].to_numpy()
    pool = np.flatnonzero((t <= 34) & pd.notna(y))
    unk_old = np.flatnonzero((t <= 34) & pd.isna(y))

    out = R / "e7d_runs.csv"
    done = set()
    if out.exists():
        done = {(r.mode, float(r.frac), int(r.seed), int(r.step)) for r in pd.read_csv(out).itertuples()}
    preds = {}
    for frac in args.fracs:
        for seed in args.seeds:
            rng = np.random.RandomState(seed)
            n_lab = max(e7c.MIN_LABELS, int(round(frac * len(pool))))
            lab = pool[rng.choice(len(pool), n_lab, replace=False)]
            yL = y[lab].astype(int)
            rest = np.setdiff1d(pool, lab)
            old_unl = np.concatenate([rest, unk_old])
            old_unl = old_unl[np.random.RandomState(seed + 7).permutation(len(old_unl))]
            static_model = None
            for step in args.steps:
                ev = np.flatnonzero((t == step) & pd.notna(y))
                for mode in ("static", "causal"):
                    if (mode, float(frac), seed, step) in done:
                        continue
                    t0 = time.time()
                    if mode == "static":
                        if static_model is None:
                            static_model = self_train(X[lab], yL, X[old_unl[:CAP]], seed)
                        m = static_model
                    else:
                        new = np.flatnonzero((t >= 35) & (t <= step))
                        if len(new) > NEW_CAP:
                            new = np.random.RandomState(seed + step).choice(new, NEW_CAP, replace=False)
                        U = np.concatenate([new, old_unl[:CAP - len(new)]])
                        m = self_train(X[lab], yL, X[U], seed)
                    p = m.predict_proba(X[ev])[:, 1]
                    yy = y[ev].astype(int)
                    row = {"mode": mode, "frac": frac, "seed": seed, "step": step, "n_eval": len(ev),
                           "n_illicit": int(yy.sum()),
                           "pr_auc": average_precision_score(yy, p) if 0 < yy.sum() < len(yy) else np.nan,
                           "seconds": round(time.time() - t0, 1)}
                    new_file = not out.exists()
                    with out.open("a", newline="", encoding="utf-8") as fh:
                        w = csv.DictWriter(fh, fieldnames=list(row))
                        if new_file:
                            w.writeheader()
                        w.writerow(row)
                    np.save(R / "e7d_preds" / f"{mode}_{frac:g}_{seed}_{step}.npy", p.astype(np.float32))
            print(f"frac={frac} seed={seed} done", flush=True)
    summarize(df)
    return 0


def summarize(df):
    """Pooled PR-AUC over periods needs the per-row predictions; recompute from saved arrays."""
    runs = pd.read_csv(R / "e7d_runs.csv")
    t = df["time_step"].to_numpy()
    y = df["y"].to_numpy()
    rows = []
    for (mode, frac, seed), g in runs.groupby(["mode", "frac", "seed"]):
        P, Y, T = [], [], []
        for step in sorted(g.step):
            f = R / "e7d_preds" / f"{mode}_{frac:g}_{seed}_{step}.npy"
            if not f.exists():
                continue
            ev = np.flatnonzero((t == step) & pd.notna(y))
            P.append(np.load(f)); Y.append(y[ev].astype(int)); T.append(np.full(len(ev), step))
        if not P:
            continue
        P, Y, T = map(np.concatenate, (P, Y, T))
        ap_ = lambda m: average_precision_score(Y[m], P[m]) if 0 < Y[m].sum() < m.sum() else np.nan
        rows.append({"mode": mode, "frac": frac, "seed": seed, "pr_auc_35_49": ap_(T > 0),
                     "pr_auc_35_42": ap_(T < 43), "pr_auc_43_49": ap_(T >= 43)})
    s = pd.DataFrame(rows)
    summ = s.groupby(["mode", "frac"]).agg(["mean", "std"]).drop(columns="seed")
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.reset_index().to_csv(R / "e7d_summary.csv", index=False, encoding="utf-8-sig")
    print(summ.round(3).to_string())

    fracs = sorted(runs.frac.unique())
    fig, axes = plt.subplots(1, len(fracs), figsize=(5 * len(fracs), 4.5), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, frac in zip(axes, fracs):
        for mode in ("static", "causal"):
            d = runs[(runs["mode"] == mode) & (runs.frac == frac)].groupby("step").pr_auc.mean()
            ax.plot(d.index, d.values, marker="o", label=mode)
        ax.axvline(43, color="black", ls="--")
        ax.set(title=f"{frac:.0%} of labels (steps 1-34)", xlabel="Test step"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("PR-AUC"); axes[0].legend()
    fig.suptitle("E7d: self-training with unlabelled transactions up to the scored step (causal) vs only 1-34 (static)")
    fig.tight_layout(); fig.savefig(F / "e7d_by_step.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
