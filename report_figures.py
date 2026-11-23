"""The figures of the setting-B report: Korean labels, at most three coloured series, one y axis.

Read-only on results/. Output: figures/fig01_*.png ... fig09_*.png
Palette: first three categorical slots (validated for colour-vision deficiency, light surface);
series are also told apart by marker and a direct label or legend, never by colour alone.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import BASE

R = BASE / "results"
F = BASE / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, GREY, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#b9b8b3", "#fcfcfb"

plt.rcParams.update({
    "font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 12,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GREY, "axes.labelcolor": MUTED,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
    "axes.axisbelow": True, "legend.frameon": False, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "lines.linewidth": 2, "lines.markersize": 8,
})


def rd(name):
    return pd.read_csv(R / f"{name}.csv", encoding="utf-8-sig")


def done(fig, name):
    fig.tight_layout()
    fig.savefig(F / name, dpi=150)
    plt.close(fig)


def share_axis(ax, xs):
    ax.set_xscale("log")
    ax.set_xticks(xs, [f"{x:g}%" for x in xs])
    ax.minorticks_off()
    ax.set_xlabel("train label ratio (log scale)")


def fig01_main():
    s = rd("eval_seed_ap").groupby("method").pr_auc.agg(["mean", "std"])
    n = rd("eval_seed_ap").groupby("method").size()
    fig, ax = plt.subplots(figsize=(9, 5))
    for key, lab, c, mk, shift, dy in (("LGBM", "LightGBM", ORANGE, "s", 1.03, -0.03),
                                       ("ST", "Self-training", BLUE, "o", 0.97, 0.025)):
        shares = [1, 5, 10, 100] if key == "LGBM" else [1, 5, 10]
        ms = [f"{key} {x}%" for x in shares]
        xs = np.array(shares) * shift
        ax.errorbar(xs, s["mean"][ms], yerr=s["std"][ms] / np.sqrt(n[ms]), color=c, marker=mk, capsize=4, label=lab)
        for x, m in zip(xs, ms):
            ax.annotate(f"{s['mean'][m]:.3f}", (x * 1.06, s["mean"][m] + dy), color=INK, fontsize=11, va="center")
    for v, lab in ((s["mean"]["BIRCH 0%"], "Unsupervised BIRCH, label 0%"), (0.065, "Random guess")):
        ax.axhline(v, color=MUTED, ls="--", lw=1.2)
        ax.text(11.5, v + 0.015, f"{lab}: {v:.3f}", color=MUTED)
    share_axis(ax, [1, 5, 10, 100])
    ax.set(ylim=(0, 0.75), ylabel="test PR-AUC (35-49)")
    ax.set_title("label 5%부터 곡선이 평평해진다")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.43))
    done(fig, "fig01_main.png")


def fig02_drift():
    d = rd("prep_B_drift").nlargest(12, "ks_train_test")[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    tp = d.feature.isin(["agg_6", "agg_7", "agg_9"]).to_numpy()
    bars = ax.barh(d.feature, d.ks_train_test, color=np.where(tp, BLUE, GREY), height=0.65)
    ax.bar_label(bars, fmt="%.2f", padding=3, color=INK)
    ax.grid(axis="x"); ax.grid(False, axis="y")
    ax.set(xlim=(0, 1.1), xlabel="train(1-30)과 test(35-49)의 distribution 차이, KS (0 = 같음, 1 = 겹치지 않음)")
    ax.set_title("block height 3개(파랑)는 train과 test가 거의 겹치지 않는다")
    done(fig, "fig02_drift.png")


def fig03_unsupervised():
    s = rd("e6b_B_summary")
    s = s[s["mode"] == "best"].sort_values("val_pr_auc", ascending=False).drop_duplicates("model").sort_values("pr_auc_mean")
    names = {"PCA-recon": "PCA reconstruction", "IForest": "Isolation Forest", "GMM-density": "GMM density",
             "GMM-component": "GMM component", "BIRCH": "BIRCH", "OCSVM": "One-Class SVM",
             "KPCA-recon": "Kernel PCA reconstruction", "HDBSCAN": "HDBSCAN"}
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.barh([names[m] for m in s.model], s.pr_auc_mean, height=0.65,
                   color=np.where((s.model == "BIRCH").to_numpy(), BLUE, GREY))
    ax.bar_label(bars, fmt="%.3f", padding=3, color=INK)
    ax.axvline(0.065, color=MUTED, ls="--", lw=1.2)
    ax.text(0.068, -0.95, "random 0.065", color=MUTED, fontsize=10)
    ax.grid(axis="x"); ax.grid(False, axis="y")
    ax.set(xlim=(0, 0.36), ylim=(-1.2, 7.6), xlabel="test PR-AUC (35-49)")
    ax.set_title("unsupervised 8종 중 BIRCH가 가장 높다")
    done(fig, "fig03_unsupervised.png")


def fig04_e7():
    g = rd("e7_summary")
    panels = [("self_training", "Self-training"), ("label_spreading", "Label Spreading"), ("cluster_rep", "Cluster representative"),
              ("cluster_prop", "Cluster label propagation"), ("active_unc", "Active learning"), ("sup_random_notp", "LightGBM, 54개 feature")]
    ref = g[g.method == "sup_random"].sort_values("frac")
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.8), sharey=True)
    for i, (ax, (k, lab)) in enumerate(zip(axes.flat, panels)):
        ax.plot(ref.frac * 100, ref.pr_auc_mean, color=GREY, marker="s", markersize=5, label="기준: LightGBM")
        d = g[g.method == k].sort_values("frac")
        ax.errorbar(d.frac * 100, d.pr_auc_mean, yerr=d.pr_auc_std, color=BLUE, marker="o", markersize=6, capsize=3,
                    label="해당 방법")
        share_axis(ax, [1, 5, 10, 50, 100])
        ax.set_xlabel("train label ratio" if i >= 3 else "")
        ax.set_title(lab)
        ax.set_ylim(0.2, 0.72)
        if i % 3 == 0:
            ax.set_ylabel("test PR-AUC")
    axes[0, 0].legend(loc="lower right", fontsize=10)
    done(fig, "fig04_e7.png")


def fig05_steps():
    m = rd("e8_partB").pivot_table(index="test_step", columns="variant", values="pr_auc")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.axvspan(42.5, 49.3, color=GRID, alpha=0.6, lw=0)
    ax.text(42.7, 1.0, "dark market 폐쇄 뒤", color=MUTED)
    for k, lab, c, mk in (("static", "Static: 1-34로 한 번 학습", ORANGE, "s"),
                          ("expanding", "매 time step 새 label로 retraining", BLUE, "o")):
        ax.plot(m.index, m[k], color=c, marker=mk, label=lab)
    ax.set_xticks(range(35, 50))
    ax.set(ylim=(0, 1.08), xlim=(34.7, 49.3), xlabel="time step (한 time step은 약 2주)", ylabel="그 time step의 PR-AUC")
    ax.set_title("retraining해도 43-45는 낮고, 47부터 회복한다")
    ax.legend(loc="lower left")
    done(fig, "fig05_steps.png")


def fig06_shift():
    s = rd("theory_shift").set_index(["period", "group"])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x, w = np.arange(2), 0.24
    old = [s.p_illicit_in_group[("1-34", g)] for g in ("single", "multi")]
    exp = [s.expected_if_prior_only[("43-49", g)] for g in ("single", "multi")]
    now = [s.p_illicit_in_group[("43-49", g)] for g in ("single", "multi")]
    for i, (v, lab, c) in enumerate(((old, "train 1-34", BLUE), (exp, "prior만 바뀌었다면", AQUA),
                                     (now, "실제 43-49", ORANGE))):
        bars = ax.bar(x + (i - 1) * (w + 0.015), np.array(v) * 100, w, color=c, label=lab)
        ax.bar_label(bars, fmt="%.1f%%", padding=3, color=INK)
    ax.set_xticks(x, ["single-input 거래", "multi-input 거래"])
    ax.grid(False, axis="x")
    ax.set(ylim=(0, 25.5), ylabel="그 group에서 illicit일 probability (%)")
    ax.set_title("폐쇄 뒤 두 group의 위험 순서가 뒤집혔다")
    ax.legend(loc="upper right")
    done(fig, "fig06_shift.png")


def fig07_recovery():
    a = rd("e8_partA")
    conds = ["train_1_42", "train_1_42_post0.1_w1", "train_1_42_post0.25_w1", "train_1_42_post0.5_w1", "train_1_42_post1_w1"]
    g = a.groupby("condition").agg(m=("pr_auc_46_49", "mean"), se=("pr_auc_46_49", "sem"), n=("n_post_labels", "mean"),
                                   lo=("n_post_illicit", "min"), hi=("n_post_illicit", "max")).loc[conds]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.errorbar(range(5), g.m, yerr=g.se, color=BLUE, marker="o", capsize=4)
    for i, v in enumerate(g.m):
        ax.annotate(f"{v:.3f}", (i + 0.08, v - 0.01), color=INK, va="top")
    ill = [f"{int(l)}-{int(h)}" if l != h else f"{int(l)}" for l, h in zip(g.lo, g.hi)]
    ax.set_xticks(range(5), [f"{int(n):,}건\n(illicit {i})" for n, i in zip(g.n, ill)])
    ax.set(ylim=(0, 0.55), xlim=(-0.3, 4.5), xlabel="더한 43-45 label 수", ylabel="46-49 PR-AUC")
    ax.set_title("폐쇄 뒤의 새 label을 더하면 score가 오른다")
    done(fig, "fig07_recovery.png")


def fig08_learning_curve():
    g = rd("theory_learning_curve").pivot_table(index="n_labels", columns="where", values="mean")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for k, lab, c, mk in (("same period", "같은 시기 1-34", AQUA, "^"), ("35-42", "미래 35-42", BLUE, "o"),
                          ("43-49", "폐쇄 뒤 43-49", ORANGE, "s")):
        ax.plot(g.index, g[k], color=c, marker=mk, label=lab)
        ax.text(g.index[-1] * 1.12, g[k].iloc[-1], f"{lab}  {g[k].iloc[-1]:.3f}", color=INK, va="center", fontsize=11)
    ax.set_xscale("log")
    ax.set_xticks(list(g.index), [f"{n:,}" for n in g.index])
    ax.minorticks_off()
    ax.set(xlim=(250, 120000), ylim=(0, 1.05), xlabel="train label 수 (log scale)", ylabel="PR-AUC")
    ax.set_title("label을 늘려도 같은 시기와 미래의 간격은 남는다")
    ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.36))
    done(fig, "fig08_learning_curve.png")


def fig09_distance():
    d = rd("theory_distance")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x, w = np.arange(len(d)), 0.3
    for i, (col, lab, c) in enumerate((("median_illicit", "illicit", ORANGE), ("median_licit", "licit", BLUE))):
        bars = ax.bar(x + (i - 0.5) * (w + 0.015), d[col], w, color=c, label=lab)
        ax.bar_label(bars, fmt="%.2f", padding=3, color=INK)
    ax.set_xticks(x, d.period)
    ax.grid(False, axis="x")
    ax.set(ylim=(0, d[["median_illicit", "median_licit"]].to_numpy().max() * 1.2), xlabel="period",
           ylabel="10번째 neighbor까지 distance (median)")
    ax.set_title("폐쇄 뒤의 illicit 거래는 train set 거래에서 멀다")
    ax.legend(loc="upper left")
    done(fig, "fig09_distance.png")


def main():
    for f in (fig01_main, fig02_drift, fig03_unsupervised, fig04_e7, fig05_steps, fig06_shift, fig07_recovery,
              fig08_learning_curve, fig09_distance):
        f()
        print("saved", f.__name__, flush=True)


if __name__ == "__main__":
    main()
