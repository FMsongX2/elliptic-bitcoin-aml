"""피처 가공 실험 (설정 B: 거래 시점 정보만).

피처 변형
- B0 time            : 거래 자체 21 + 선행 거래 집계 36 (E2 설정 B)
- B1 -height         : B0에서 선행 거래 블록 높이 6개(사실상 날짜) 제거
- B2 +shape          : B1 + 거래 모양 (입출력 개수, 입력1·출력1 등 플래그, 비율)
- B3 +pred           : B2 + 선행 거래들의 모양
- B4 pct only        : 원래 값 대신 직전 time step 기준 백분위 + shape + pred
- B5 +pct            : B3 + 백분위 (원래 값과 백분위 함께)

모델
- LightGBM 기본      : E2와 같은 설정 (트리 500개, 잎 31개)
- LightGBM 규제      : 트리 200개, 잎 8개, 잎당 최소 100건, 학습률 0.05 (과적합 억제)
- 결정 트리          : 깊이 1~8 중 검증 F1이 가장 높은 깊이

절차 (팀 공통 규칙)
1) 1~30 학습 → 31~34(검증)에서 F1 최대 임계값과 트리 깊이 선택
2) 같은 설정으로 1~34 재학습 → 35~49 평가
최종 비교는 **검증 F1**로 고르고, 테스트 F1은 보고만 한다.

실행: python fe_experiment.py
결과: results/fe_summary.csv, figures/fe_f1_by_variant.png
"""
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

from common import (BASE, DARK_MARKET_STEP, NBR_IN, TX, clear_results, evaluate, load_dataset,
                    load_edges, log_result, split, tune_threshold)
from feature_engineering import (HEIGHT_COLS, PCT_PREFIX, PRED_PREFIX, SHAPE_PREFIX,
                                 build_engineered_features)

warnings.filterwarnings("ignore", category=UserWarning)

EXPERIMENT = "FE"
SEEDS = [0, 1, 2]
TREE_DEPTHS = range(1, 9)
FIG_DIR = BASE / "figures"
RES_DIR = BASE / "results"


def variants(fe_cols):
    shape = [c for c in fe_cols if c.startswith(SHAPE_PREFIX)]
    pred = [c for c in fe_cols if c.startswith(PRED_PREFIX)]
    pct = [c for c in fe_cols if c.startswith(PCT_PREFIX)]
    b0 = TX + NBR_IN
    b1 = [c for c in b0 if c not in HEIGHT_COLS]
    return {
        "B0 time": b0,
        "B1 -height": b1,
        "B2 +shape": b1 + shape,
        "B3 +pred": b1 + shape + pred,
        "B4 pct only": pct + shape + pred,
        "B5 +pct": b1 + shape + pred + pct,
    }


def make_models(seed):
    return {
        "LightGBM 기본": lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.5, random_state=seed, n_jobs=-1, verbose=-1),
        "LightGBM 규제": lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=8, min_child_samples=100,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0,
            random_state=seed, n_jobs=-1, verbose=-1),
    }


def fit_predict(model, train, test, cols):
    return model.fit(train[cols], train["y"]).predict_proba(test[cols])[:, 1]


def run_tree(train, val, trainval, test, cols):
    """검증 F1로 깊이를 고르고, 그 깊이로 1~34 재학습."""
    best = None
    for d in TREE_DEPTHS:
        pv = fit_predict(DecisionTreeClassifier(max_depth=d, random_state=0), train, val, cols)
        th = tune_threshold(val["y"], pv)
        vf = f1_score(val["y"], pv >= th)
        if best is None or vf > best[0]:
            best = (vf, d, th)
    vf, d, th = best
    prob = fit_predict(DecisionTreeClassifier(max_depth=d, random_state=0), trainval, test, cols)
    return vf, th, prob, f"depth={d}"


def summarize(test, prob, th):
    y = test["y"].values
    early = test["time_step"].values < DARK_MARKET_STEP
    m = evaluate(y, prob, th)
    return {**m, "f1_35_42": f1_score(y[early], prob[early] >= th),
            "f1_43_49": f1_score(y[~early], prob[~early] >= th)}


def main():
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
    clear_results(EXPERIMENT)

    df_all = load_dataset(labeled_only=False)
    fe = build_engineered_features(df_all, load_edges())
    df = df_all[df_all["y"].notna()].merge(fe, on="txId")
    df["y"] = df["y"].astype(int)
    train, val, test = split(df)
    trainval = pd.concat([train, val])

    rows = []
    for vname, cols in variants([c for c in fe.columns if c != "txId"]).items():
        for seed in SEEDS:
            for mname, model in make_models(seed).items():
                pv = fit_predict(model, train, val, cols)
                th = tune_threshold(val["y"], pv)
                vf = f1_score(val["y"], pv >= th)
                prob = fit_predict(make_models(seed)[mname], trainval, test, cols)
                s = summarize(test, prob, th)
                log_result(EXPERIMENT, mname, vname, seed, {k: s[k] for k in ["precision", "recall", "f1", "pr_auc"]},
                           "setting=B, threshold=val")
                rows.append({"variant": vname, "model": mname, "seed": seed, "n_features": len(cols),
                             "val_f1": vf, **s})
        vf, th, prob, note = run_tree(train, val, trainval, test, cols)
        s = summarize(test, prob, th)
        log_result(EXPERIMENT, "결정 트리", vname, 0, {k: s[k] for k in ["precision", "recall", "f1", "pr_auc"]},
                   f"setting=B, threshold=val, {note}")
        rows.append({"variant": vname, "model": "결정 트리", "seed": 0, "n_features": len(cols),
                     "val_f1": vf, **s, "note": note})
        print(f"  done: {vname} ({len(cols)} features), 트리 {note}")

    res = pd.DataFrame(rows)
    summary = (res.groupby(["variant", "model"], sort=False)
               .agg(n_features=("n_features", "first"), val_f1=("val_f1", "mean"),
                    test_f1=("f1", "mean"), test_f1_std=("f1", "std"),
                    precision=("precision", "mean"), recall=("recall", "mean"),
                    pr_auc=("pr_auc", "mean"), f1_35_42=("f1_35_42", "mean"),
                    f1_43_49=("f1_43_49", "mean"), note=("note", "first"))
               .round(4).reset_index())
    summary.to_csv(RES_DIR / "fe_summary.csv", index=False, encoding="utf-8-sig")
    plot(summary)

    best = summary.loc[summary["val_f1"].idxmax()]
    print("\n=== 피처 가공 결과 (설정 B, LightGBM은 시드 3개 평균) ===")
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(summary.drop(columns=["note"]).to_string(index=False))
    print(f"\n검증 F1 기준 최종 선택: {best['variant']} / {best['model']} "
          f"(검증 {best['val_f1']:.3f}, 테스트 F1 {best['test_f1']:.3f}, PR-AUC {best['pr_auc']:.3f})")


def plot(summary):
    models = summary["model"].unique()
    names = summary["variant"].unique()
    colors = {"LightGBM 기본": "#2a78d6", "LightGBM 규제": "#1baf7a", "결정 트리": "#eb6834"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    x = np.arange(len(names))
    w = 0.26
    for ax, col, title in [(axes[0], "val_f1", "검증 F1 (31~34) — 선택 기준"),
                           (axes[1], "test_f1", "테스트 F1 (35~49) — 보고용")]:
        for i, m in enumerate(models):
            vals = summary[summary["model"] == m].set_index("variant").loc[names, col]
            bars = ax.bar(x + (i - 1) * w, vals, w, color=colors[m], label=m)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
        ax.set_xticks(x, names, rotation=20, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e6e5e0")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("불법 F1")
    axes[1].legend(loc="upper right", frameon=False)
    fig.suptitle("[설정 B] 피처 가공 변형별 성능", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fe_f1_by_variant.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
