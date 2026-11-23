"""Fill every table of the setting-B report from results/ (no hand-copied numbers).

In the report a generated table sits between two comment lines:
    <!-- table:NAME -->
    ...
    <!-- /table -->
Usage (repo root):  python report_tables.py          rewrite the tables in place
                    python report_tables.py --check  exit 1 if any table differs from results/

Judgement rule: "higher" or "lower" only if |difference| > t(0.975, seeds - 1) seed standard errors and,
where a 95% interval from resampling time steps exists, that interval excludes 0; otherwise undecided.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parent
R = ROOT / "results"
MARK = re.compile(r"<!-- table:(\w+) -->\r?\n(.*?)<!-- /table -->", re.S)
PCTS = [0.01, 0.05, 0.10, 0.50, 1.0]
BASE_ST = dict(kind="st", base="team", half_life=None, unsup=False, thr=0.95, balanced=False, plw=1.0)
METHOD = {"BIRCH": "Unsupervised BIRCH", "LGBM": "LightGBM", "ST": "Self-training",
          "ENS-past": "Ensemble (past-only rank)", "ENS-prob": "Ensemble (soft voting)"}


def rd(name, **kw):
    return pd.read_csv(R / f"{name}.csv", encoding="utf-8-sig", **kw)


def f3(v):
    return "-" if pd.isna(v) else f"{v:.3f}"


def signed(v):
    text = f"{v:+.3f}"
    return "0.000" if text in ("+0.000", "-0.000") else text


def pct(v, digits=1):
    return f"{100 * v:.{digits}f}%"


def md(head, rows):
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def name(method):
    key, share = method.rsplit(" ", 1)
    return f"{METHOD[key]} {share}"


def crit(n):
    return float(student_t.ppf(0.975, n - 1))


def verdict(r):
    sure = abs(r["diff"]) > crit(6) * r.seed_se
    if sure and r.ci95_low_steps > 0:
        return "**높다**"
    if sure and r.ci95_high_steps < 0:
        return "**낮다**"
    return "판정 불가"


# ---------------------------------------------------------------- headline and final evaluation
def headline():
    s = rd("eval_seed_ap").groupby("method")[["pr_auc", "pr_auc_35_42", "pr_auc_43_49"]].mean()
    b = rd("eval_bootstrap").set_index("method")
    order = ["BIRCH 0%"] + [f"{m} {p}" for p in ("1%", "5%", "10%") for m in ("LGBM", "ST")] + ["LGBM 100%"]
    rows = [["Random guess", "-", "0.065", "-", "0.092", "0.025"]]
    for m in order:
        key, share = m.split(" ")
        rows.append([METHOD[key], share, f"**{s.pr_auc[m]:.3f}**",
                     f"{b.ci95_low_steps[m]:.3f}-{b.ci95_high_steps[m]:.3f}", f3(s.pr_auc_35_42[m]),
                     f3(s.pr_auc_43_49[m])])
    return md(["방법", "train label", "PR-AUC 35-49", "95% CI", "폐쇄 전 35-42", "폐쇄 뒤 43-49"], rows)


def _paired(pairs):
    p = rd("eval_paired").set_index(["a", "b"])
    rows = []
    for a, b in pairs:
        r = p.loc[(a, b)]
        rows.append([f"{name(a)} - {name(b)}", f"{signed(r['diff'])} ± {r.seed_se:.3f}",
                     f"{r.ci95_low_steps:.3f}에서 {r.ci95_high_steps:.3f}", verdict(r)])
    return md(["비교 (앞 - 뒤)", "차이 ± standard error", "95% CI", "판정"], rows)


def st_vs_lgbm():
    return _paired([(f"ST {p}", f"LGBM {p}") for p in ("1%", "5%", "10%")])


def few_vs_all():
    return _paired([(f"{m} {p}", "LGBM 100%") for m in ("LGBM", "ST") for p in ("1%", "5%", "10%")])


def ensemble_scores():
    s = rd("eval_seed_ap").groupby("method").pr_auc.mean()
    rows = [[METHOD[m]] + [f3(s[f"{m} {p}"]) for p in ("1%", "5%", "10%")]
            for m in ("LGBM", "ST", "ENS-past", "ENS-prob")]
    return md(["방법", "1%", "5%", "10%"], rows)


def ensemble_vs_st():
    return _paired([(f"ENS-{rule} {p}", f"ST {p}") for rule in ("past", "prob") for p in ("1%", "5%", "10%")])


def topk():
    t = rd("eval_topk").set_index(["method", "period", "k"])
    rows = []
    for m in ("BIRCH 0%", "LGBM 1%", "ST 1%", "LGBM 10%", "ST 10%", "ENS-past 10%", "ENS-prob 10%", "LGBM 100%"):
        pre, post = t.loc[(m, "35-42", 100)], t.loc[(m, "43-49", 100)]
        rows.append([name(m), f3(pre.recall_at_k), f3(pre.precision_at_k), f3(post.recall_at_k), f3(post.precision_at_k)])
    return md(["방법", "35-42 recall", "35-42 precision", "43-49 recall", "43-49 precision"], rows)


def calibration():
    rows = [[name(r.method), f3(r.ece), f"{r.mean_pred_35_42:.3f} / {r.illicit_share_35_42:.3f}",
             f"{r.mean_pred_43_49:.3f} / {r.illicit_share_43_49:.3f}"] for r in rd("eval_calibration").itertuples()]
    return md(["model", "ECE", "35-42 예측 / 실제", "43-49 예측 / 실제"], rows)


# ---------------------------------------------------------------- data and preprocessing
def diagnostics():
    s, d = rd("prep_B_feature_stats"), rd("prep_B_drift").set_index("feature")
    normal = ((s.skewness.abs() < 0.5) & (s.excess_kurtosis.abs() < 1)).sum()
    tp = d.loc[["agg_6", "agg_7", "agg_9"], "ks_train_test"]
    return md(["진단 (train 1-30)", "값"], [
        ["kurtosis median (normal distribution은 0)", f"{s.excess_kurtosis.median():.0f}"],
        ["normal distribution에 가까운 feature", f"{normal} / {len(s)}"],
        ["값의 절반 넘게 한 값에 몰린 feature", f"{(s.most_frequent_value_share > 0.5).sum()} / {len(s)}"],
        ["5 standard deviation을 넘는 outlier가 있는 feature", f"{(s.fraction_abs_z_gt_5 > 0).sum()} / {len(s)}"],
        ["time-proxy feature(agg_6/7/9)의 train-test KS", f"{tp.min():.3f}-{tp.max():.3f}"],
    ])


def prep_compare():
    q = rd("prep_research_B_summary").set_index("pipeline")
    label = {"std_57": "Standardization (57개)", "quantile_57": "Quantile transform (57개)", "quantile_54": "Quantile transform (54개)",
             "log_54": "Log transform (54개)", "yeojohnson_54": "Yeo-Johnson (54개)", "split_54": "Mode-value split (54개)",
             "split_dedup_54": "Mode-value split + deduplication"}
    cols = ["knn_0.01", "knn_0.1", "knn_1", "label_spreading_0.01", "label_spreading_0.1"]
    rows = []
    for k, v in label.items():
        rows.append([v] + [f"**{q.loc[k, c]:.3f}**" if q.loc[k, c] == q[c].max() else f3(q.loc[k, c]) for c in cols])
    return md(["preprocessing", "kNN 1%", "kNN 10%", "kNN 100%", "Label Spreading 1%", "Label Spreading 10%"], rows)


def embedding():
    e = rd("e7b_summary").set_index(["method", "embedding", "frac"]).pr_auc_mean
    ml = {"label_spreading": "Label Spreading", "cluster_rep": "Cluster representative", "cluster_prop": "Cluster label propagation"}
    el = {"quantile_57": "Quantile transform 57개", "quantile_54": "Quantile transform 54개", "yeojohnson_54": "Yeo-Johnson 54개"}
    rows = [[mv, ev] + [f3(e.get((m, k, f), np.nan)) for f in PCTS] for m, mv in ml.items() for k, ev in el.items()]
    return md(["방법", "distance space", "1%", "5%", "10%", "50%", "100%"], rows)


# ---------------------------------------------------------------- E6b, E7, E7c/d
def unsupervised():
    s = rd("e6b_B_summary")
    s = s[s["mode"] == "best"].sort_values("val_pr_auc", ascending=False).drop_duplicates("model")
    label = {"PCA-recon": "PCA reconstruction", "IForest": "Isolation Forest", "GMM-density": "GMM density",
             "GMM-component": "GMM component", "BIRCH": "BIRCH", "OCSVM": "One-Class SVM",
             "KPCA-recon": "Kernel PCA reconstruction", "HDBSCAN": "HDBSCAN"}
    rows = [[label[r.model], r.prep, f3(r.val_pr_auc), f"**{r.pr_auc_mean:.3f}**", f3(r.f1_mean),
             f3(r.precision_mean), f3(r.recall_mean), f3(r.pr_auc_35_42_mean), f3(r.pr_auc_43_49_mean)]
            for r in s.sort_values("pr_auc_mean", ascending=False).itertuples()]
    return md(["detector", "preprocessing", "validation", "test", "F1", "precision", "recall", "35-42", "43-49"], rows)


E7_LABEL = {"sup_random": "LightGBM", "sup_random_notp": "LightGBM (54개 feature)", "self_training": "Self-training",
            "label_spreading": "Label Spreading", "cluster_rep": "Cluster representative", "cluster_prop": "Cluster label propagation",
            "active_unc": "Active learning"}


def e7_prauc():
    g = rd("e7_summary").set_index(["method", "frac"])
    rows = [[v] + [f"{g.pr_auc_mean[(k, f)]:.3f} ± {g.pr_auc_std[(k, f)]:.3f}" if (k, f) in g.index else "-"
                   for f in PCTS] for k, v in E7_LABEL.items()]
    return md(["방법", "1%", "5%", "10%", "50%", "100%"], rows)


def e7_f1():
    g = rd("e7_summary").set_index(["method", "frac"])
    rows = []
    for k in ("sup_random", "self_training", "label_spreading", "cluster_rep"):
        rows.append([E7_LABEL[k]] + [f"{g.f1_mean[(k, f)]:.3f} ({g.precision_mean[(k, f)]:.2f} / {g.recall_mean[(k, f)]:.2f})"
                                     for f in (0.01, 0.05, 0.10)])
    return md(["방법", "1%", "5%", "10%"], rows)


def threshold():
    g = rd("e7b_summary")
    g = g[g.embedding == "-"].set_index(["method", "frac"])
    rows = [[E7_LABEL[m], lab] + [f3(g[col][(m, f)]) for f in PCTS]
            for m in ("sup_random", "self_training")
            for col, lab in (("f1_full_mean", "validation label 전부"), ("f1_real_mean", "같은 ratio만"))]
    return md(["방법", "threshold를 정한 label", "1%", "5%", "10%", "50%", "100%"], rows)


def lgbm_variants():
    c = rd("e7c_runs")
    c = c[c.split == "test"]
    c = c.join(pd.DataFrame([json.loads(x) for x in c.config], index=c.index))
    s = c[c.kind == "sup"].copy()
    s["half_life"] = s.half_life.fillna(0).astype(int)
    out = []
    for col, title, levels in (
            ("base", "Base hyperparameter", [("team", "team config"), ("small", "small"), ("balanced", "balanced"), ("shallow", "shallow")]),
            ("half_life", "최근 거래 weight", [(0, "없음"), (20, "half-life 20"), (10, "half-life 10"), (5, "half-life 5")]),
            ("unsup", "unsupervised score feature", [(False, "없음"), (True, "있음")])):
        m = s.groupby(col).pr_auc.mean()
        out.append([title, ", ".join(f"{lab} {m[k]:.3f}" for k, lab in levels)])
    return md(["바꾼 것", "test PR-AUC (1/5/10% mean)"], out)


def st_variants():
    c = rd("e7c_runs")
    t = c[c.split == "test"].pivot_table(index=["frac", "seed"], columns="config", values="pr_auc")
    cfgs = {k: json.loads(k) for k in t}
    base = next(k for k, v in cfgs.items() if v == BASE_ST)
    d = t.sub(t[base], axis=0)
    rows = []
    for k, v in cfgs.items():
        if v["kind"] != "st" or v["base"] != "team" or v["unsup"] or v["half_life"] or k == base:
            continue
        parts = [f"illicit {v['thr_pos']} / licit {v['thr']}" if "thr_pos" in v else f"threshold {v['thr']}"]
        parts += ["class balancing"] * v["balanced"] + [f"weight {v['plw']}"] * (v["plw"] != 1)
        cells = []
        for f in (0.01, 0.05, 0.10):
            x = d[k].xs(f, level="frac")
            cells.append(f"{signed(x.mean())} ± {x.sem():.3f}")
        rows.append((v["balanced"], ", ".join(parts), cells))
    rows.sort(key=lambda r: (r[0], r[1]))
    return md(["variant (기본은 threshold 0.95)", "1%", "5%", "10%"], [[p] + c for _, p, c in rows])


def test_time_st():
    g = rd("e7d_summary").set_index(["mode", "frac"])
    rows = []
    for f in (0.01, 0.05, 0.10):
        s, c = g.loc[("static", f)], g.loc[("causal", f)]
        rows.append([pct(f, 0), f3(s.pr_auc_35_49_mean), f3(c.pr_auc_35_49_mean), f3(s.pr_auc_43_49_mean),
                     f3(c.pr_auc_43_49_mean)])
    return md(["label", "35-49 static", "35-49 adaptive", "43-49 static", "43-49 adaptive"], rows)


def selection():
    j = json.loads((R / "e7c_selection.json").read_text("utf-8"))
    return md(["hyperparameter config를 고르는 규칙", "test ranking과의 correlation", "고른 hyperparameter config의 test PR-AUC"], [
        ["validation 31-34 하나", f"{j['spearman_single_vs_test']:.2f}", f3(j["pick_single"]["test_mean"])],
        ["Rolling-origin validation 3개 mean", f"{j['spearman_rolling_vs_test']:.2f}", f3(j["pick_rolling"]["test_mean"])]])


# ---------------------------------------------------------------- shift, E8, E9
def shift():
    s = rd("theory_shift").set_index(["period", "group"])
    rows = []
    for p in ("1-34", "35-42", "43-49"):
        a, b = s.loc[(p, "single")], s.loc[(p, "multi")]
        rows.append([p, pct(a.share_of_illicit), pct(a.share_of_licit), pct(a.p_illicit_in_group), pct(b.p_illicit_in_group)])
    return md(["period", "illicit 중 single-input", "licit 중 single-input", "single-input의 illicit probability", "multi-input의 illicit probability"], rows)


def recovery():
    a = rd("e8_partA")
    base = a[a.condition == "train_1_42"].set_index("seed").pr_auc_46_49
    rows = []
    for cond, lab in (("train_1_34", "1-34"), ("train_1_42", "1-42 (기준)"),
                      ("train_1_42_post0.1_w1", "1-42 + 43-45의 10%"), ("train_1_42_post0.25_w1", "1-42 + 43-45의 25%"),
                      ("train_1_42_post0.5_w1", "1-42 + 43-45의 50%"), ("train_1_42_post1_w1", "1-42 + 43-45 전부"),
                      ("train_43_45", "43-45만")):
        g = a[a.condition == cond].set_index("seed")
        d, n = g.pr_auc_46_49 - base, g.n_post_illicit
        ill = f"{n.min()}-{n.max()}" if n.min() != n.max() else f"{n.min()}"
        if cond == "train_1_42":
            change, judge = "-", "-"
        else:
            change = f"{signed(d.mean())} ± {d.sem():.3f}"
            judge = "판정 불가" if abs(d.mean()) <= crit(len(d)) * d.sem() else ("**높다**" if d.mean() > 0 else "**낮다**")
        rows.append([lab, f"{g.n_post_labels.iloc[0]:,}", ill, f3(g.pr_auc_46_49.mean()), change, judge])
    return md(["training data", "새 label", "그중 illicit", "46-49 PR-AUC", "기준과의 차이", "판정"], rows)


def rolling():
    m = rd("e8_partB").pivot_table(index="test_step", columns="variant", values="pr_auc")
    lab = {"static": "Static (1-34)", "expanding": "매 time step retraining (expanding window)", "window10": "매 time step retraining (sliding window 10)"}
    rows = [[v] + [f3(m.loc[a:b, k].mean()) for a, b in ((35, 42), (43, 45), (47, 49), (43, 49))] for k, v in lab.items()]
    return md(["Training 방식", "35-42", "43-45", "47-49", "43-49"], rows)


def shift_ssl():
    r = rd("e8b_runs")
    g = r.groupby(["post_frac", "method"]).pr_auc_46_49.mean()
    piv = r.pivot_table(index=["post_frac", "seed"], columns="method", values="pr_auc_46_49")
    rows = []
    for f in (0.0, 0.1, 0.25, 0.5):
        x = (piv.ST - piv.LGBM).xs(f, level="post_frac")
        rows.append([f"{int(r[r.post_frac == f].n_post_labels.iloc[0]):,}"] +
                    [f3(g[(f, m)]) for m in ("LGBM", "ST", "LS", "ENS")] + [f"{signed(x.mean())} ± {x.sem():.3f}"])
    return md(["새 label", "LightGBM", "Self-training", "Label Spreading", "Ensemble (soft voting)", "Self-training - LightGBM"], rows)


def no_label_fixes():
    s = rd("e9_summary").set_index(["train", "eval", "method"]).pr_auc
    h = rd("e9_hedge").set_index(["train", "eval"])
    cols = [("1-34", "35-42"), ("1-34", "43-49"), ("1-34", "35-49"), ("1-42", "43-49")]
    lab = [("global", "Baseline model"), ("group_rank", "Within-group ranking"), ("group_em", "EM prior adjustment"),
           ("group_em_cal", "Calibration + EM"), ("group_em_shrink", "EM shrinkage"), ("group_em_rel", "Reference-prior EM"),
           ("no_input", "input-structure feature 제거")]
    rows = [[v] + [f3(s[(a, b, k)]) for a, b in cols] for k, v in lab]
    rows += [[v] + [f3(h.loc[c, k]) for c in cols]
             for k, v in (("hedge_past", "Hedge (past-only rank)"), ("hedge_prob", "Hedge (soft voting)"))]
    return md(["방법", "35-42", "43-49", "35-49 전체", "train 1-42, test 43-49"], rows)


# ---------------------------------------------------------------- theory
def learning_curve():
    g = rd("theory_learning_curve").pivot_table(index=["frac", "n_labels"], columns="where", values=["mean", "roc_auc"])
    rows = [[f"{n:,}"] + [f"{r[('mean', w)]:.3f} / {r[('roc_auc', w)]:.3f}" for w in ("same period", "35-42", "43-49")]
            for (f, n), r in g.iterrows()]
    return md(["train label 수", "같은 시기 (1-34)", "미래 35-42", "폐쇄 뒤 43-49"], rows)


def pseudo():
    g = rd("theory_pseudo_labels").set_index(["frac", "where"])
    place = {"same period": "같은 시기", "35-42": "미래 35-42", "43-49": "폐쇄 뒤 43-49"}
    rows = []
    for f in (0.01, 0.10):
        for k, v in place.items():
            r = g.loc[(f, k)]
            rows.append([pct(f, 0), v, pct(r.precision_confident_illicit), pct(r.illicit_confidently_called_licit)])
    return md(["model의 label", "Evaluation period", "illicit으로 확신한 것 중 실제 illicit", "실제 illicit 중 licit으로 확신"], rows)


def label_shift():
    s = rd("theory_shift").set_index(["period", "group"])
    rows = []
    for g, lab in (("single", "single-input"), ("multi", "multi-input")):
        o, n = s.loc[("1-34", g)], s.loc[("43-49", g)]
        rows.append([lab, pct(o.p_illicit_in_group), pct(n.expected_if_prior_only), f"**{pct(n.p_illicit_in_group)}**"])
    return md(["group", "1-34 illicit probability", "prior만 바뀌었을 때 예상", "43-49 실제"], rows)


def error_corr():
    e = rd("theory_ensemble")
    rows = [[pct(r.frac, 0), f"{r.error_corr_ST_LGBM:.2f}", f"{r.error_corr_ST_LS:.2f}", f"{r.error_corr_LGBM_LS:.2f}",
             f3(r.pr_auc_LS)] for r in e.itertuples()]
    return md(["label", "Self-training과 LightGBM", "Self-training과 Label Spreading", "LightGBM과 Label Spreading", "Label Spreading 단독 PR-AUC"], rows)


def roc_auc():
    r = rd("theory_roc_auc").set_index(["method", "frac", "period"]).roc_auc
    rows = [[f"{METHOD[m]} {pct(f, 0)}", f3(r[(m, f, "35-42")]), f3(r[(m, f, "43-49")])]
            for f in (0.01, 0.10) for m in ("LGBM", "ST")]
    return md(["방법", "ROC-AUC 35-42", "ROC-AUC 43-49"], rows)


def distance():
    rows = [[r.period, f"{r.median_illicit:.2f}", f"{r.median_licit:.2f}", f3(r.roc_auc_distance_to_illicit)]
            for r in rd("theory_distance").itertuples()]
    return md(["period", "illicit", "licit", "distance로 illicit을 맞히는 ROC-AUC"], rows)


TABLES = {f.__name__: f for f in (
    headline, st_vs_lgbm, few_vs_all, ensemble_scores, ensemble_vs_st, topk, calibration, diagnostics, prep_compare,
    embedding, unsupervised, e7_prauc, e7_f1, threshold, lgbm_variants, st_variants, test_time_st, selection, shift,
    recovery, rolling, shift_ssl, no_label_fixes, learning_curve, pseudo, label_shift, error_corr, roc_auc, distance)}


def report_path():
    return ROOT / "docs" / "02_설정B_unsupervised_semi_supervised.md"


def fill(text):
    return MARK.sub(lambda m: f"<!-- table:{m.group(1)} -->\n\n{TABLES[m.group(1)]()}\n<!-- /table -->", text)


def main(argv):
    p = report_path()
    text = p.read_bytes().decode("utf-8").replace("\r\n", "\n")
    new = fill(text)
    if "--check" in argv:
        print("tables up to date" if new == text else "tables differ from results/")
        return int(new != text)
    p.write_bytes(new.encode("utf-8"))
    print(f"filled {len(MARK.findall(new))} tables in {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
