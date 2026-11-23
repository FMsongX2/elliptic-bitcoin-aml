"""E7c: search for better label-efficient methods on setting B (57 transaction-time features).

Selection protocol (new): rolling-origin (forward-chaining) validation inside steps 1-34
  fold 1: fit 1-20 -> validate 21-24
  fold 2: fit 1-26 -> validate 27-30
  fold 3: fit 1-30 -> validate 31-34   (the single split used by E6b/E7)
Selection score = mean validation PR-AUC over folds, label shares and seeds.
Test (fit 1-34 -> 35-49) is computed for every configuration but used only to analyse how well
each selection rule transfers (rank correlation), never to pick.

Label shares: 1%, 5%, 10% of the labelled rows of the fit steps (at least 20).

Stage 1, supervised LightGBM variants (32 configs):
  base        team    E2 parameters (500 trees, lr 0.05, 31 leaves, colsample 0.5, subsample 0.8)
              small   300 trees, 15 leaves, min_child_samples 5, reg_lambda 1
              balanced team + scale_pos_weight = licit/illicit among revealed labels
              shallow 800 trees, lr 0.03, 7 leaves, min_child_samples 10
  half_life   None / 5 / 10 / 20 steps: sample weight 0.5 ** ((t_max - t) / half_life)
  unsup       add two label-free scores fitted on all rows of the fit steps:
              PCA reconstruction error (5 comps) and BIRCH centroid distance (E6b-B best),
              both on standardised features without agg_6/7/9
Stage 2, self-training variants on top of a stage-1 config (custom loop so weights can be used):
  threshold 0.9 / 0.95 / 0.99, class-balanced pseudo labels on/off, pseudo-label weight 1 / 0.5,
  class-specific thresholds (illicit 0.7 / 0.85, licit 0.95; FlexMatch-style, Zhang et al., 2021),
  5 rounds, unlabelled pool capped at 30,000 rows. Plus label_spreading (E7, quantile_57).
Ensembles of the stored predictions are evaluated in eval_reliability.py, with rules that never
look at a later time step.

Outputs: results/e7c_runs.csv (one row per config x split x frac x seed, resumable),
results/e7c_preds.joblib (stored probabilities), results/e7c_summary.csv.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from common import BASE
import common
from common import feature_cols, load_dataset
import e7_semi_supervised as e7
from detectors_e6b import BirchDetector, PCAReconDetector

RESULTS_DIR = BASE / "results"
TIME_PROXY = ["agg_6", "agg_7", "agg_9"]
FOLDS = [("cv1", range(1, 21), range(21, 25)), ("cv2", range(1, 27), range(27, 31)),
         ("cv3", range(1, 31), range(31, 35))]
TEST = ("test", range(1, 35), range(35, 50))
FRACS = [0.01, 0.05, 0.10]
SEL_SEEDS = [0, 1]
TEST_SEEDS = [0, 1, 2]
MIN_LABELS = 20
ST_UNLABELED_CAP = 30000

BASES = {
    "team": dict(n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                 colsample_bytree=0.5),
    "small": dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=5, reg_lambda=1.0,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.5),
    "balanced": dict(n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                     colsample_bytree=0.5),
    "shallow": dict(n_estimators=800, learning_rate=0.03, num_leaves=7, min_child_samples=10, subsample=0.8,
                    subsample_freq=1, colsample_bytree=0.5),
}


def config_id(cfg):
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"))


def stage1_configs():
    return [dict(kind="sup", base=b, half_life=h, unsup=u)
            for b in BASES for h in (None, 5, 10, 20) for u in (False, True)]


def stage2_configs(base_cfg):
    out = [dict(kind="st", base="team", half_life=None, unsup=False, thr=0.95, balanced=False, plw=1.0)]  # E7-like
    for thr in (0.9, 0.95, 0.99):
        for bal in (False, True):
            out.append(dict(base_cfg, kind="st", thr=thr, balanced=bal, plw=1.0))
    for bal in (False, True):  # down-weighted pseudo labels only at the loosest threshold
        out.append(dict(base_cfg, kind="st", thr=0.9, balanced=bal, plw=0.5))
    # class-specific thresholds (FlexMatch / FreeMatch style): accept illicit pseudo labels earlier
    for thr_pos in (0.7, 0.85):
        for bal in (False, True):
            out.append(dict(base_cfg, kind="st", thr=0.95, thr_pos=thr_pos, balanced=bal, plw=1.0))
    out.append(dict(kind="ls"))
    seen, uniq = set(), []
    for c in out:
        if config_id(c) not in seen:
            seen.add(config_id(c))
            uniq.append(c)
    return uniq


# ---------------------------------------------------------------- data helpers
class Data:
    def __init__(self, df):
        self.cols = feature_cols(df, "time")
        self.notp = np.array([c not in TIME_PROXY for c in self.cols])
        self.X = df[self.cols].to_numpy(np.float32)
        self.t = df["time_step"].to_numpy()
        self.y = df["y"].to_numpy()
        self._unsup = {}

    def split(self, fit_steps, eval_steps):
        fit = np.flatnonzero(np.isin(self.t, list(fit_steps)))
        ev = np.flatnonzero(np.isin(self.t, list(eval_steps)) & pd.notna(self.y))
        pool = fit[pd.notna(self.y[fit])]
        unk = fit[pd.isna(self.y[fit])]
        return fit, pool, unk, ev

    def unsup_scores(self, key, fit_idx):
        """Two label-free scores fitted on all rows of the fit steps; returns a function rows -> (n, 2)."""
        if key not in self._unsup:
            Xs = self.X[fit_idx][:, self.notp]
            sc = StandardScaler().fit(Xs)
            Z = sc.transform(Xs)
            pca = PCAReconDetector(n_components=5, seed=0).fit(Z)
            birch = BirchDetector(threshold=1.0, n_clusters=8, pca=20, seed=0).fit(Z)

            def f(rows, sc=sc, pca=pca, birch=birch):
                Zr = sc.transform(self.X[rows][:, self.notp])
                return np.column_stack([np.log1p(pca.score(Zr)), np.log1p(birch.score(Zr))]).astype(np.float32)
            self._unsup[key] = f
        return self._unsup[key]


def lgbm(base, seed, y_lab=None):
    params = dict(BASES[base])
    if base == "balanced" and y_lab is not None and y_lab.sum() > 0:
        params["scale_pos_weight"] = float((len(y_lab) - y_lab.sum()) / y_lab.sum())
    return lgb.LGBMClassifier(random_state=seed, n_jobs=-1, verbose=-1, **params)


def recency(t, t_max, half_life):
    """Sample weight halving every `half_life` steps before the latest fit step."""
    if half_life is None:
        return np.ones(len(t))
    return 0.5 ** ((t_max - t) / float(half_life))


# ---------------------------------------------------------------- one run
def run(cfg, D, split, frac, seed):
    """Train cfg on the fit steps of `split` with `frac` of the labels; return probabilities on eval rows."""
    name, fit_steps, eval_steps = split
    fit, pool, unk, ev = D.split(fit_steps, eval_steps)
    rng = np.random.RandomState(seed)
    n_lab = max(MIN_LABELS, int(round(frac * len(pool))))
    q = rng.choice(len(pool), min(n_lab, len(pool)), replace=False)
    lab = pool[q]
    y_lab = D.y[lab].astype(int)

    if cfg["kind"] == "ls":  # E7 label spreading; embedding fitted on the fit-step rows only
        Xf = D.X[fit]
        if name not in _LS:
            emb = e7.Embedding(seed=0).fit(Xf)
            _LS[name] = {"emb": emb, "Z": emb.transform(Xf)}
        pool_l, unk_l = np.searchsorted(fit, pool), np.searchsorted(fit, unk)
        model, n_l, n_i = e7.train_method("label_spreading", Xf, pool_l, D.y[pool].astype(int), unk_l,
                                          n_lab, seed, dict(_LS[name]))
        return model.predict_proba(D.X[ev])[:, 1], ev, n_l, n_i

    feats = (lambda rows: D.X[rows])
    if cfg["unsup"]:
        us = D.unsup_scores(name, fit)
        feats = (lambda rows, us=us: np.hstack([D.X[rows], us(rows)]))
    t_max = D.t[fit].max()
    w_lab = recency(D.t[lab], t_max, cfg["half_life"])

    if len(np.unique(y_lab)) < 2:
        return np.full(len(ev), y_lab.mean()), ev, len(lab), int(y_lab.sum())

    if cfg["kind"] == "sup":
        m = lgbm(cfg["base"], seed, y_lab).fit(feats(lab), y_lab, sample_weight=w_lab)
        return m.predict_proba(feats(ev))[:, 1], ev, len(lab), int(y_lab.sum())

    # self-training with weights
    rest = np.setdiff1d(pool, lab)
    U = np.concatenate([rest, unk])
    if len(U) > ST_UNLABELED_CAP:
        U = rng.choice(U, ST_UNLABELED_CAP, replace=False)
    XL, yL, wL = feats(lab), y_lab.copy(), w_lab.copy()
    XU = feats(U)
    wU = recency(D.t[U], t_max, cfg["half_life"])
    neg_per_pos = (len(y_lab) - y_lab.sum()) / max(1, y_lab.sum())
    alive = np.ones(len(U), bool)
    m = None
    for _ in range(5):
        m = lgbm(cfg["base"], seed, yL).fit(XL, yL, sample_weight=wL)
        if not alive.any():
            break
        p = m.predict_proba(XU[alive])[:, 1]
        idx = np.flatnonzero(alive)
        pos = idx[p >= cfg.get("thr_pos", cfg["thr"])]
        neg_mask = p <= 1 - cfg["thr"]
        neg = idx[neg_mask]
        if cfg["balanced"]:
            k = int(round(len(pos) * neg_per_pos))
            neg = idx[neg_mask][np.argsort(p[neg_mask])[:k]]
        add = np.concatenate([pos, neg])
        if len(add) == 0:
            break
        XL = np.vstack([XL, XU[add]])
        yL = np.concatenate([yL, np.ones(len(pos), int), np.zeros(len(neg), int)])
        wL = np.concatenate([wL, cfg["plw"] * wU[add]])
        alive[add] = False
    else:  # round limit reached: the last pseudo labels still need a fit (as in sklearn)
        m = lgbm(cfg["base"], seed, yL).fit(XL, yL, sample_weight=wL)
    return m.predict_proba(feats(ev))[:, 1], ev, len(lab), int(y_lab.sum())


_LS = {}  # split name -> {"emb", "Z"} for label spreading


# ---------------------------------------------------------------- driver
FIELDS = ["config", "stage", "split", "frac", "seed", "n_labels", "n_illicit_labels", "pr_auc",
          "pr_auc_35_42", "pr_auc_43_49", "seconds"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=[1, 2])
    ap.add_argument("--base-config", default=None, help="stage 2: JSON of the stage-1 config to build on")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args(argv)

    df = load_dataset(labeled_only=False)
    if args.fast:
        keep = df.sample(frac=1.0, random_state=0).groupby("time_step").head(150).index
        df = df.loc[np.sort(keep)].reset_index(drop=True)
    D = Data(df)
    runs_csv = RESULTS_DIR / "e7c_runs.csv"
    preds_path = RESULTS_DIR / "e7c_preds.joblib"
    done = set()
    if runs_csv.exists() and not args.fast:
        done = {(r.config, r.split, float(r.frac), int(r.seed)) for r in pd.read_csv(runs_csv).itertuples()}
    preds = joblib.load(preds_path) if preds_path.exists() and not args.fast else {}

    cfgs = stage1_configs() if args.stage == 1 else stage2_configs(json.loads(args.base_config))
    jobs = [(c, s, f, sd) for c in cfgs for s in FOLDS for f in FRACS for sd in SEL_SEEDS] + \
           [(c, TEST, f, sd) for c in cfgs for f in FRACS for sd in TEST_SEEDS]
    print(f"stage {args.stage}: {len(cfgs)} configs, {len(jobs)} runs", flush=True)
    n_new = 0
    for i, (cfg, split, frac, seed) in enumerate(jobs):
        cid = config_id(cfg)
        if (cid, split[0], float(frac), seed) in done:
            continue
        t0 = time.time()
        p, ev, n_lab, n_ill = run(cfg, D, split, frac, seed)
        y_ev = D.y[ev].astype(int)
        row = {"config": cid, "stage": args.stage, "split": split[0], "frac": frac, "seed": seed,
               "n_labels": n_lab, "n_illicit_labels": n_ill, "pr_auc": average_precision_score(y_ev, p),
               "pr_auc_35_42": np.nan, "pr_auc_43_49": np.nan, "seconds": round(time.time() - t0, 1)}
        if split[0] == "test":
            te = D.t[ev]
            for nm, mk in [("pr_auc_35_42", te < 43), ("pr_auc_43_49", te >= 43)]:
                row[nm] = average_precision_score(y_ev[mk], p[mk]) if len(np.unique(y_ev[mk])) == 2 else np.nan
        preds[(cid, split[0], float(frac), seed)] = p.astype(np.float32)
        new = not runs_csv.exists()
        with runs_csv.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
        n_new += 1
        if n_new % 25 == 0:
            joblib.dump(preds, preds_path)
            print(f"  {i + 1}/{len(jobs)} runs ({row['seconds']}s last)", flush=True)
    joblib.dump(preds, preds_path)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
