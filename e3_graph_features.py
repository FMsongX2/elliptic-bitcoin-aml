"""E3: 직접 만든 그래프 피처를 LightGBM에 추가했을 때의 효과.

피처셋 (모두 LightGBM, 시드 3개, 1~34 학습 / 35~49 테스트, 임계값 0.5)
- local                      : 로컬 93 (E2 기준)
- local+struct               : + 구조 피처 9개
- local+nbr                  : + 방향별 이웃 평균 186개
- local+struct+nbr+nbr2      : 기존 집계 72개 대신 우리가 만든 그래프 피처만 사용
- all                        : 로컬 93 + 기존 집계 72 (E2 최고 모델)
- all+struct                 : + 구조 피처
- all+struct+nbr+nbr2        : 전부

E2에서 검증 구간 임계값 조정은 효과가 없었으므로 임계값은 0.5로 고정한다.
실행: python e3_graph_features.py
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (BASE, DARK_MARKET_STEP, clear_results, evaluate, load_dataset, load_edges,
                    log_result, split)
from e2_baseline import make_model
from graph_features import NBR2_PREFIX, NBR_PREFIXES, STRUCT_PREFIX, build_graph_features

warnings.filterwarnings("ignore", category=UserWarning)

EXPERIMENT = "E3"
SEEDS = [0, 1, 2]
FIG_DIR = BASE / "figures"
RES_DIR = BASE / "results"

GROUP_COLORS = {"local": "#2a78d6", "agg": "#1baf7a", "struct": "#eb6834",
                "nbr": "#e87ba4", "nbr2": "#6250d6"}
GROUP_LABELS = {"local": "로컬 93", "agg": "기존 이웃 집계 72", "struct": "구조 (E3)",
                "nbr": "방향별 이웃 평균 (E3)", "nbr2": "2-hop 이웃 평균 (E3)"}


def feature_group(col):
    if col.startswith("local_"):
        return "local"
    if col.startswith("agg_"):
        return "agg"
    if col.startswith(STRUCT_PREFIX):
        return "struct"
    if col.startswith(NBR_PREFIXES):
        return "nbr"
    if col.startswith(NBR2_PREFIX):
        return "nbr2"
    raise ValueError(col)


def variant_columns(df):
    groups = {g: [c for c in df.columns if c != "txId" and not c.startswith(("time_step", "y"))
                  and _safe_group(c) == g] for g in GROUP_COLORS}
    return {
        "local": groups["local"],
        "local+struct": groups["local"] + groups["struct"],
        "local+nbr": groups["local"] + groups["nbr"],
        "local+struct+nbr+nbr2": groups["local"] + groups["struct"] + groups["nbr"] + groups["nbr2"],
        "all": groups["local"] + groups["agg"],
        "all+struct": groups["local"] + groups["agg"] + groups["struct"],
        "all+struct+nbr+nbr2": sum(groups.values(), []),
    }


def _safe_group(col):
    try:
        return feature_group(col)
    except ValueError:
        return None


def period_f1(test, prob):
    early = test["time_step"].values < DARK_MARKET_STEP
    y = test["y"].values
    return (evaluate(y[early], prob[early])["f1"], evaluate(y[~early], prob[~early])["f1"])


def plot_variants(summary):
    order = summary.sort_values("f1_mean").index
    fig, ax = plt.subplots(figsize=(9, 4.8))
    base = summary.loc["all", "f1_mean"]
    colors = ["#eb6834" if "struct" in v or "nbr" in v else "#2a78d6" for v in order]
    ax.barh(order, summary.loc[order, "f1_mean"], xerr=summary.loc[order, "f1_std"],
            color=colors, height=0.6, edgecolor="white")
    for i, v in enumerate(order):
        val = summary.loc[v, "f1_mean"]
        ax.text(val + summary.loc[v, "f1_std"] + 0.003, i, f"{val:.3f}", va="center", fontsize=9,
                color="#0b0b0b")
    ax.axvline(base, color="#52514e", linestyle="--", linewidth=1)
    ax.text(base, len(order) - 0.45, " E2 최고 (all)", color="#52514e", fontsize=9)
    lo = summary["f1_mean"].min() - 0.05
    ax.set_xlim(lo, summary["f1_mean"].max() + 0.04)
    ax.set_xlabel("테스트 불법 F1 (35~49, 시드 3개 평균)")
    ax.set_title("피처셋별 LightGBM 성능 (주황 = E3 그래프 피처 포함)", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e6e5e0")
    ax.set_axisbelow(True)
    fig.savefig(FIG_DIR / "e3_f1_by_variant.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_importance(imp):
    by_group = imp.groupby("group")["gain"].sum()
    by_group = (by_group / by_group.sum()).sort_values()
    top = imp.sort_values("gain", ascending=False).head(20).iloc[::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1, 1.3]})
    ax1.barh([GROUP_LABELS[g] for g in by_group.index], by_group.values,
             color=[GROUP_COLORS[g] for g in by_group.index], height=0.6)
    for i, v in enumerate(by_group.values):
        ax1.text(v + 0.005, i, f"{v:.1%}", va="center", fontsize=9)
    ax1.set_xlim(0, by_group.max() + 0.08)
    ax1.set_title("피처 그룹별 중요도 비중 (gain)", loc="left", fontweight="bold")

    ax2.barh(top["feature"], top["gain"] / imp["gain"].sum(),
             color=[GROUP_COLORS[g] for g in top["group"]], height=0.7)
    ax2.set_title("상위 20개 피처 (색 = 그룹)", loc="left", fontweight="bold")
    ax2.tick_params(axis="y", labelsize=8)
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#e6e5e0")
        ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[g]) for g in GROUP_COLORS]
    ax2.legend(handles, [GROUP_LABELS[g] for g in GROUP_COLORS], loc="lower right",
               fontsize=8, frameon=False)
    fig.suptitle("LightGBM 피처 중요도 (all+struct+nbr+nbr2, seed 0)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e3_feature_importance.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
    clear_results(EXPERIMENT)

    df_all = load_dataset(labeled_only=False)
    gf = build_graph_features(df_all, load_edges())
    df = df_all[df_all["y"].notna()].merge(gf, on="txId")
    df["y"] = df["y"].astype(int)
    train, val, test = split(df)
    trainval = pd.concat([train, val])
    variants = variant_columns(df)

    rows, imp = [], None
    for name, cols in variants.items():
        for seed in SEEDS:
            model = make_model("LightGBM", seed).fit(trainval[cols], trainval["y"])
            prob = model.predict_proba(test[cols])[:, 1]
            m = evaluate(test["y"], prob)
            log_result(EXPERIMENT, "LightGBM", name, seed, m, "threshold=0.5")
            f1_early, f1_late = period_f1(test, prob)
            rows.append({"variant": name, "n_features": len(cols), "seed": seed, **m,
                         "f1_35_42": f1_early, "f1_43_49": f1_late})
            if name == "all+struct+nbr+nbr2" and seed == 0:
                imp = pd.DataFrame({"feature": cols, "gain": model.booster_.feature_importance("gain")})
                imp["group"] = imp["feature"].map(feature_group)
        print(f"  done: {name} ({len(cols)} features)")

    res = pd.DataFrame(rows)
    summary = res.groupby("variant").agg(
        n_features=("n_features", "first"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        precision=("precision", "mean"), recall=("recall", "mean"),
        pr_auc=("pr_auc", "mean"),
        f1_35_42=("f1_35_42", "mean"), f1_43_49=("f1_43_49", "mean"),
    ).loc[list(variants)].round(4)
    summary.to_csv(RES_DIR / "e3_summary.csv", encoding="utf-8-sig")
    imp.sort_values("gain", ascending=False).to_csv(RES_DIR / "e3_feature_importance.csv",
                                                    index=False, encoding="utf-8-sig")

    plot_variants(summary)
    plot_importance(imp)

    print("\n=== E3 테스트 결과 (LightGBM, 시드 3개 평균, 임계값 0.5) ===")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary)


if __name__ == "__main__":
    main()
