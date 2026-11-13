"""E2: 지도학습 베이스라인.

모델 {Logistic Regression, Random Forest, LightGBM} x 피처셋 {local 93, all 165}.
1) train(1~30)으로 학습 → val(31~34)에서 불법 F1 최대 임계값 탐색
2) 같은 설정으로 train+val(1~34) 재학습 → test(35~49) 평가
   임계값 0.5 결과와 조정 임계값 결과를 모두 기록한다.

실행: python e2_baseline.py
"""
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (BASE, DARK_MARKET_STEP, clear_results, evaluate, evaluate_by_step,
                    feature_cols, load_dataset, log_result, split, tune_threshold)

warnings.filterwarnings("ignore", category=UserWarning)

EXPERIMENT = "E2"
SEEDS = [0, 1, 2]
FIG_DIR = BASE / "figures"


def make_model(name, seed):
    if name == "LogReg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=500, max_features=50, n_jobs=-1, random_state=seed)
    if name == "LightGBM":
        return lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                                  random_state=seed, n_jobs=-1, verbose=-1)
    raise ValueError(name)


def run_one(name, feature_set, seed, train, val, test):
    cols = feature_cols(train, feature_set)

    model = make_model(name, seed).fit(train[cols], train["y"])
    threshold = tune_threshold(val["y"], model.predict_proba(val[cols])[:, 1])

    trainval = pd.concat([train, val])
    model = make_model(name, seed).fit(trainval[cols], trainval["y"])
    prob = model.predict_proba(test[cols])[:, 1]

    for th, note in [(0.5, "threshold=0.5"), (threshold, "threshold=tuned_on_val")]:
        log_result(EXPERIMENT, name, feature_set, seed, evaluate(test["y"], prob, th), note)
    return prob, threshold


def plot_by_step(step_frames):
    colors = {"LogReg": "#1baf7a", "RandomForest": "#2a78d6", "LightGBM": "#eb6834"}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.1})
    for name, sdf in step_frames.items():
        ax1.plot(sdf["time_step"], sdf["f1"], marker="o", markersize=4, linewidth=2,
                 color=colors[name], label=name)
    any_df = next(iter(step_frames.values()))
    ax2.bar(any_df["time_step"], any_df["n_illicit"], color="#c3c2b7", width=0.7)

    for ax in (ax1, ax2):
        ax.axvline(DARK_MARKET_STEP - 0.5, color="#52514e", linestyle="--", linewidth=1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e6e5e0")
        ax.set_axisbelow(True)
    ax1.text(DARK_MARKET_STEP - 0.3, 0.95, "다크마켓 폐쇄", color="#52514e", fontsize=9)
    ax1.set_ylim(-0.03, 1.03)
    ax1.set_ylabel("불법 F1")
    ax1.set_title("테스트 구간 time step별 불법 F1 (피처 all, seed 0, 검증 임계값)",
                  loc="left", fontweight="bold")
    ax1.legend(loc="lower left", frameon=False)
    ax2.set_ylabel("불법 거래 수")
    ax2.set_xlabel("time step")
    fig.savefig(FIG_DIR / "e2_f1_by_step.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
    clear_results(EXPERIMENT)
    df = load_dataset(labeled_only=True)
    train, val, test = split(df)
    print(f"train {len(train):,} / val {len(val):,} / test {len(test):,} (라벨 있는 거래)")

    step_frames = {}
    for name in ["LogReg", "RandomForest", "LightGBM"]:
        seeds = [0] if name == "LogReg" else SEEDS
        for feature_set in ["local", "all"]:
            for seed in seeds:
                prob, th = run_one(name, feature_set, seed, train, val, test)
                if feature_set == "all" and seed == 0:
                    step_frames[name] = evaluate_by_step(test, prob, th)
            print(f"  done: {name} / {feature_set}")

    FIG_DIR.mkdir(exist_ok=True)
    plot_by_step(step_frames)
    pd.concat(step_frames, names=["model"]).reset_index(level=0).to_csv(
        BASE / "results" / "e2_by_step.csv", index=False, encoding="utf-8-sig")

    res = pd.read_csv(BASE / "results" / "results.csv", encoding="utf-8-sig")
    res = res[res["experiment"] == EXPERIMENT]
    summary = (res.groupby(["model", "feature_set", "note"])[["precision", "recall", "f1", "pr_auc"]]
               .agg(["mean", "std"]).round(3))
    print("\n=== E2 테스트 결과 (seed 평균, 표준편차) ===")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary)


if __name__ == "__main__":
    main()
