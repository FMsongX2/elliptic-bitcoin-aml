"""팀 공통 모듈: 데이터 로드, 시간 기반 분할, 평가, 결과 기록.

모든 실험(E1~E8)은 이 모듈의 함수만 사용해 같은 기준으로 결과를 쌓는다.

    from common import load_dataset, split, feature_cols, evaluate, tune_threshold, log_result
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data" / "elliptic"
RESULTS_PATH = BASE / "results" / "results.csv"

SEED = 42

# 시간 기반 분할 (원 논문 방식: 학습 1~34 / 테스트 35~49).
# 하이퍼파라미터·임계값은 학습 구간 끝부분(31~34)을 검증셋으로 떼어 정한다.
TRAIN_STEPS = range(1, 31)
VAL_STEPS = range(31, 35)
TEST_STEPS = range(35, 50)
DARK_MARKET_STEP = 43


def load_dataset(labeled_only=True):
    """features + 라벨을 합친 DataFrame. y: 1=불법, 0=합법, NaN=unknown."""
    features = pd.read_csv(DATA_DIR / "elliptic_txs_features.csv", header=None)
    features.columns = (["txId", "time_step"]
                        + [f"local_{i}" for i in range(93)]
                        + [f"agg_{i}" for i in range(72)])
    classes = pd.read_csv(DATA_DIR / "elliptic_txs_classes.csv")
    classes["y"] = classes["class"].map({"1": 1, "2": 0})
    df = features.merge(classes[["txId", "y"]], on="txId")
    if labeled_only:
        df = df[df["y"].notna()].copy()
        df["y"] = df["y"].astype(int)
    return df


def load_edges():
    return pd.read_csv(DATA_DIR / "elliptic_txs_edgelist.csv")


# 피처 의미(Šafář et al., 2026; feature_semantics.csv)에 따른 묶음
TX = [f"local_{i}" for i in range(21)]            # 거래 자체 (금액, 수수료, 입출력 수 등)
ADDR = [f"local_{i}" for i in range(21, 93)]      # 입출력 주소 누적 통계 (데이터 수집 종료 시점 기준 → 미래 포함)
NBR_IN = [f"agg_{i}" for i in range(36)]          # 선행 거래 집계 (과거)
NBR_OUT = [f"agg_{i}" for i in range(36, 72)]     # 후속 거래 집계 (미래)

# 평가 설정
#   A = 전체 165개 (기존 연구와 같은 설정, 미래 정보 포함)
#   B = 거래 시점 57개 (거래가 일어난 순간에 알 수 있는 정보만: 거래 자체 + 선행 거래 집계)
FEATURE_SETS = {
    "local": TX + ADDR,
    "all": TX + ADDR + NBR_IN + NBR_OUT,   # 설정 A
    "time": TX + NBR_IN,                   # 설정 B
}


def feature_cols(df, feature_set):
    """'local' = 로컬 93개, 'all' = 165개(설정 A), 'time' = 거래 시점 57개(설정 B)."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature_set: {feature_set}")
    return FEATURE_SETS[feature_set]


def z_to_count(z):
    """정규화된 개수 피처(입력 수, 출력 수 등)를 실제 개수로 되돌린다.

    값의 간격이 일정하므로 (z - 최솟값) / 간격 + 1 이 개수다(최솟값 = 1개).
    """
    u = np.unique(z)
    step = np.min(np.diff(u[:10]))
    return np.rint((z - u[0]) / step).astype(int) + 1


def split(df):
    """(train, val, test). 최종 모델은 train+val(1~34)로 다시 학습한다."""
    ts = df["time_step"]
    return (df[ts.isin(TRAIN_STEPS)], df[ts.isin(VAL_STEPS)], df[ts.isin(TEST_STEPS)])


def tune_threshold(y_true, prob):
    """검증셋에서 불법 F1을 최대화하는 임계값."""
    candidates = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y_true, prob >= t, zero_division=0) for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def evaluate(y_true, prob, threshold=0.5):
    """불법 클래스(1) 기준 지표."""
    pred = (prob >= threshold).astype(int)
    return {
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, prob),
        "threshold": threshold,
    }


def evaluate_by_step(test_df, prob, threshold=0.5):
    """time step별 지표 (E8 시간 강건성 분석용)."""
    rows = []
    for step, idx in test_df.groupby("time_step").indices.items():
        y = test_df["y"].values[idx]
        m = evaluate(y, prob[idx], threshold)
        rows.append({"time_step": step, "n_illicit": int(y.sum()), **m})
    return pd.DataFrame(rows)


def clear_results(experiment):
    """재실행 시 중복을 막기 위해 해당 실험의 기존 기록을 지운다."""
    if RESULTS_PATH.exists():
        res = pd.read_csv(RESULTS_PATH, encoding="utf-8-sig")
        res[res["experiment"] != experiment].to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")


def log_result(experiment, model, feature_set, seed, metrics, note=""):
    """results/results.csv 에 한 줄 추가. 팀 전체가 같은 파일·형식을 쓴다."""
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"),
           "experiment": experiment, "model": model, "feature_set": feature_set,
           "seed": seed, **{k: round(v, 4) for k, v in metrics.items()}, "note": note}
    pd.DataFrame([row]).to_csv(RESULTS_PATH, mode="a", index=False,
                               header=not RESULTS_PATH.exists(), encoding="utf-8-sig")
