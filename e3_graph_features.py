"""E3: 직접 만든 그래프 피처를 LightGBM에 추가했을 때의 효과.

설정 A (전체 165개, 미래 정보 포함) — build_graph_features (앞뒤 연결 모두)
- local                      : 로컬 93 (E2 기준)
- local+struct               : + 구조 피처 9개
- local+nbr                  : + 방향별 이웃 평균 186개
- local+struct+nbr+nbr2      : 기존 집계 72개 대신 우리가 만든 그래프 피처만 사용
- all                        : 로컬 93 + 기존 집계 72 (E2 최고 모델)
- all+struct                 : + 구조 피처
- all+struct+nbr+nbr2        : 전부

설정 B (거래 시점 57개) — build_past_graph_features (선행 거래 쪽 연결만)
- time                       : 거래 자체 21 + 선행 거래 집계 36 (E2 설정 B)
- time+pstruct               : + 과거 구조 피처 5개
- time+pnbr                  : + 1·2단계 조상 거래의 거래 자체 피처 평균 42개
- time+pstruct+pnbr          : 전부

모두 LightGBM, 시드 3개, 1~34 학습 / 35~49 테스트, 임계값 0.5.
E2에서 검증 구간 임계값 조정은 효과가 없었으므로 임계값은 0.5로 고정한다.
실행: python e3_graph_features.py
"""
import warnings

import matplotlib.pyplot as plt
import pandas as pd

from common import (ADDR, BASE, DARK_MARKET_STEP, NBR_IN, NBR_OUT, TX, clear_results, evaluate,
                    load_dataset, load_edges, log_result, split)
from e2_baseline import make_model
from graph_features import (NBR2_PREFIX, NBR_PREFIXES, PAST_NBR_PREFIXES, PAST_STRUCT_PREFIX,
                            STRUCT_PREFIX, build_graph_features, build_past_graph_features)

warnings.filterwarnings("ignore", category=UserWarning)

EXPERIMENT = "E3"
SEEDS = [0, 1, 2]
FIG_DIR = BASE / "figures"
RES_DIR = BASE / "results"

GROUP_COLORS = {"tx": "#2a78d6", "addr": "#86b6ef", "nbr_in": "#1baf7a", "nbr_out": "#8fd6b8",
                "struct": "#eb6834", "nbr": "#e87ba4", "nbr2": "#6250d6",
                "pstruct": "#eb6834", "pnbr": "#e87ba4"}
GROUP_LABELS = {"tx": "거래 자체 21", "addr": "주소 통계 72 (미래 포함)",
                "nbr_in": "선행 거래 집계 36", "nbr_out": "후속 거래 집계 36 (미래)",
                "struct": "구조 (E3)", "nbr": "방향별 이웃 평균 (E3)", "nbr2": "2-hop 이웃 평균 (E3)",
                "pstruct": "과거 구조 (E3-B)", "pnbr": "조상 거래 평균 (E3-B)"}

_TX, _ADDR, _NBR_IN, _NBR_OUT = set(TX), set(ADDR), set(NBR_IN), set(NBR_OUT)


def feature_group(col):
    if col in _TX:
        return "tx"
    if col in _ADDR:
        return "addr"
    if col in _NBR_IN:
        return "nbr_in"
    if col in _NBR_OUT:
        return "nbr_out"
    if col.startswith(STRUCT_PREFIX):
        return "struct"
    if col.startswith(NBR_PREFIXES):
        return "nbr"
    if col.startswith(NBR2_PREFIX):
        return "nbr2"
    if col.startswith(PAST_STRUCT_PREFIX):
        return "pstruct"
    if col.startswith(PAST_NBR_PREFIXES):
        return "pnbr"
    raise ValueError(col)


def variant_columns(df):
    """{변형 이름: (설정, 피처 목록)}"""
    groups = {g: [] for g in GROUP_COLORS}
    for c in df.columns:
        if c not in ("txId", "time_step", "y"):
            groups[feature_group(c)].append(c)
    local = TX + ADDR
    allf = TX + ADDR + NBR_IN + NBR_OUT
    time = TX + NBR_IN
    return {
        "local": ("A", local),
        "local+struct": ("A", local + groups["struct"]),
        "local+nbr": ("A", local + groups["nbr"]),
        "local+struct+nbr+nbr2": ("A", local + groups["struct"] + groups["nbr"] + groups["nbr2"]),
        "all": ("A", allf),
        "all+struct": ("A", allf + groups["struct"]),
        "all+struct+nbr+nbr2": ("A", allf + groups["struct"] + groups["nbr"] + groups["nbr2"]),
        "time": ("B", time),
        "time+pstruct": ("B", time + groups["pstruct"]),
        "time+pnbr": ("B", time + groups["pnbr"]),
        "time+pstruct+pnbr": ("B", time + groups["pstruct"] + groups["pnbr"]),
    }


def period_f1(test, prob):
    early = test["time_step"].values < DARK_MARKET_STEP
    y = test["y"].values
    return (evaluate(y[early], prob[early])["f1"], evaluate(y[~early], prob[~early])["f1"])


def plot_variants(summary, base, title, fname):
    order = summary.sort_values("f1_mean").index
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(order) + 1))
    colors = ["#eb6834" if "+" in v else "#2a78d6" for v in order]
    ax.barh(order, summary.loc[order, "f1_mean"], xerr=summary.loc[order, "f1_std"],
            color=colors, height=0.6, edgecolor="white")
    for i, v in enumerate(order):
        val = summary.loc[v, "f1_mean"]
        ax.text(val + summary.loc[v, "f1_std"] + 0.003, i, f"{val:.3f}", va="center", fontsize=9,
                color="#0b0b0b")
    base_val = summary.loc[base, "f1_mean"]
    ax.axvline(base_val, color="#52514e", linestyle="--", linewidth=1)
    ax.text(base_val, len(order) - 0.45, f" E2 기준 ({base})", color="#52514e", fontsize=9)
    ax.set_xlim(summary["f1_mean"].min() - 0.05, summary["f1_mean"].max() + 0.04)
    ax.set_xlabel("테스트 불법 F1 (35~49, 시드 3개 평균)")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e6e5e0")
    ax.set_axisbelow(True)
    fig.savefig(FIG_DIR / fname, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_importance(imp, title, fname):
    by_group = imp.groupby("group")["gain"].sum()
    by_group = (by_group / by_group.sum()).sort_values()
    top = imp.sort_values("gain", ascending=False).head(20).iloc[::-1]
    present = [g for g in GROUP_COLORS if g in set(imp["group"])]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1, 1.3]})
    ax1.barh([GROUP_LABELS[g] for g in by_group.index], by_group.values,
             color=[GROUP_COLORS[g] for g in by_group.index], height=0.6)
    for i, v in enumerate(by_group.values):
        ax1.text(v + 0.005, i, f"{v:.1%}", va="center", fontsize=9)
    ax1.set_xlim(0, by_group.max() + 0.1)
    ax1.set_title("피처 그룹별 중요도 비중 (gain)", loc="left", fontweight="bold")

    ax2.barh(top["feature"], top["gain"] / imp["gain"].sum(),
             color=[GROUP_COLORS[g] for g in top["group"]], height=0.7)
    ax2.set_title("상위 20개 피처 (색 = 그룹)", loc="left", fontweight="bold")
    ax2.tick_params(axis="y", labelsize=8)
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#e6e5e0")
        ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[g]) for g in present]
    ax2.legend(handles, [GROUP_LABELS[g] for g in present], loc="lower right", fontsize=8, frameon=False)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
    clear_results(EXPERIMENT)

    df_all = load_dataset(labeled_only=False)
    edges = load_edges()
    gf = build_graph_features(df_all, edges)
    pf = build_past_graph_features(df_all, edges)
    df = df_all[df_all["y"].notna()].merge(gf, on="txId").merge(pf, on="txId")
    df["y"] = df["y"].astype(int)
    train, val, test = split(df)
    trainval = pd.concat([train, val])
    variants = variant_columns(df)
    importance_for = {"all+struct+nbr+nbr2": "A", "time+pstruct+pnbr": "B"}

    rows, imps = [], {}
    for name, (setting, cols) in variants.items():
        for seed in SEEDS:
            model = make_model("LightGBM", seed).fit(trainval[cols], trainval["y"])
            prob = model.predict_proba(test[cols])[:, 1]
            m = evaluate(test["y"], prob)
            log_result(EXPERIMENT, "LightGBM", name, seed, m, f"setting={setting}, threshold=0.5")
            f1_early, f1_late = period_f1(test, prob)
            rows.append({"variant": name, "setting": setting, "n_features": len(cols), "seed": seed,
                         **m, "f1_35_42": f1_early, "f1_43_49": f1_late})
            if name in importance_for and seed == 0:
                imp = pd.DataFrame({"feature": cols, "gain": model.booster_.feature_importance("gain")})
                imp["group"] = imp["feature"].map(feature_group)
                imps[importance_for[name]] = imp
        print(f"  done: [{setting}] {name} ({len(cols)} features)")

    res = pd.DataFrame(rows)
    summary = res.groupby("variant").agg(
        setting=("setting", "first"),
        n_features=("n_features", "first"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        precision=("precision", "mean"), recall=("recall", "mean"),
        pr_auc=("pr_auc", "mean"),
        f1_35_42=("f1_35_42", "mean"), f1_43_49=("f1_43_49", "mean"),
    ).loc[list(variants)].round(4)
    summary.to_csv(RES_DIR / "e3_summary.csv", encoding="utf-8-sig")
    for setting, imp in imps.items():
        suffix = "" if setting == "A" else "_B"
        imp.sort_values("gain", ascending=False).to_csv(RES_DIR / f"e3_feature_importance{suffix}.csv",
                                                        index=False, encoding="utf-8-sig")

    plot_variants(summary[summary["setting"] == "A"], "all",
                  "[설정 A] 피처셋별 LightGBM 성능 (주황 = E3 그래프 피처 포함)", "e3_f1_by_variant.png")
    plot_variants(summary[summary["setting"] == "B"], "time",
                  "[설정 B] 피처셋별 LightGBM 성능 (주황 = 과거 그래프 피처 포함)", "e3_f1_by_variant_B.png")
    plot_importance(imps["A"], "[설정 A] LightGBM 피처 중요도 (all+struct+nbr+nbr2, seed 0)",
                    "e3_feature_importance.png")
    plot_importance(imps["B"], "[설정 B] LightGBM 피처 중요도 (time+pstruct+pnbr, seed 0)",
                    "e3_feature_importance_B.png")

    print("\n=== E3 테스트 결과 (LightGBM, 시드 3개 평균, 임계값 0.5) ===")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary)
    for setting, imp in imps.items():
        share = imp.groupby("group")["gain"].sum()
        print(f"[{setting}] 그룹별 중요도:", (share / share.sum()).round(3).sort_values(ascending=False).to_dict())


if __name__ == "__main__":
    main()
