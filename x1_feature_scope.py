"""X1: 피처 범위(scope)별 기여도와 폐쇄 이후 구간의 순위·임계값 분해.

Šafář et al. (2026)가 복원한 피처 의미(feature_semantics.csv)에 따라 165개 피처를 나눈다.
  tx      : 거래 자체 정보 (local_0~20, 21개). 거래 시점에 계산 가능
  addr    : 입출력 주소의 누적 통계 (local_21~92, 72개). 블록 575,059 시점 기준으로 계산되어
            테스트 구간 이후의 활동까지 포함한다 (시간 누수, 주소 단위 비독립)
  nbr_in  : 선행 거래 집계 (agg_0~35). 거래 시점에 계산 가능
  nbr_out : 후속 거래 집계 (agg_36~71). 해당 거래 이후에 생기는 정보

같은 LightGBM 설정(E2)으로 1~34 학습, 35~49 평가한다. 기간별로 F1(임계값 0.5)뿐 아니라
PR-AUC, ROC-AUC, 사후 최적 임계값 F1을 함께 기록해 순위 실패와 임계값 실패를 구분한다.

실행: python x1_feature_scope.py
결과: results/x1_feature_scope.csv
"""
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from common import BASE, DARK_MARKET_STEP, clear_results, evaluate, load_dataset, log_result, split

warnings.filterwarnings("ignore", category=UserWarning)

EXPERIMENT = "X1"
SEEDS = [0, 1, 2]

TX = [f"local_{i}" for i in range(21)]
ADDR = [f"local_{i}" for i in range(21, 93)]
NBR_IN = [f"agg_{i}" for i in range(36)]
NBR_OUT = [f"agg_{i}" for i in range(36, 72)]

VARIANTS = {
    "tx": TX,
    "addr": ADDR,
    "nbr_in": NBR_IN,
    "nbr_out": NBR_OUT,
    "tx+nbr_in": TX + NBR_IN,
    "tx+nbr_in+nbr_out": TX + NBR_IN + NBR_OUT,
    "local(tx+addr)": TX + ADDR,
    "all": TX + ADDR + NBR_IN + NBR_OUT,
}


def make_model(seed):
    return lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                              random_state=seed, n_jobs=8, verbose=-1)


def best_f1(y, prob):
    """라벨을 보고 고른 임계값에서의 F1. 순위만으로 도달 가능한 상한이다."""
    precision, recall, _ = precision_recall_curve(y, prob)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(f1.max())


def period_metrics(y, prob):
    m = evaluate(y, prob, 0.5)
    return {"f1": m["f1"], "recall": m["recall"], "pr_auc": m["pr_auc"],
            "roc_auc": roc_auc_score(y, prob), "best_f1": best_f1(y, prob),
            "prevalence": float(np.mean(y))}


def main():
    df = load_dataset(labeled_only=True)
    train, val, test = split(df)
    trainval = pd.concat([train, val])
    y = test["y"].values
    ts = test["time_step"].values
    periods = {"35_49": np.ones(len(test), bool),
               "35_42": ts < DARK_MARKET_STEP,
               "43_49": ts >= DARK_MARKET_STEP}

    clear_results(EXPERIMENT)
    rows = []
    for name, cols in VARIANTS.items():
        per_seed = []
        for seed in SEEDS:
            prob = make_model(seed).fit(trainval[cols], trainval["y"]).predict_proba(test[cols])[:, 1]
            log_result(EXPERIMENT, "LightGBM", name, seed, evaluate(y, prob, 0.5), "threshold=0.5")
            per_seed.append({f"{k}_{p}": v for p, mask in periods.items()
                             for k, v in period_metrics(y[mask], prob[mask]).items()})
        mean = pd.DataFrame(per_seed).mean()
        rows.append({"variant": name, "n_features": len(cols), **mean.round(4).to_dict()})
        print(f"{name:20s} n={len(cols):3d}  F1 {mean['f1_35_49']:.3f}  PR-AUC {mean['pr_auc_35_49']:.3f}"
              f" | 35~42 F1 {mean['f1_35_42']:.3f}"
              f" | 43~49 F1 {mean['f1_43_49']:.3f} PR-AUC {mean['pr_auc_43_49']:.3f}"
              f" ROC-AUC {mean['roc_auc_43_49']:.3f} bestF1 {mean['best_f1_43_49']:.3f}"
              f" (prev {mean['prevalence_43_49']:.3f})", flush=True)

    out = BASE / "results" / "x1_feature_scope.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
