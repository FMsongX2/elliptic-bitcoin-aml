"""X4: 주소 단위에서 누수 있는 피처와 없는 피처의 비교 (Elliptic++ actors).

Liang et al. (2026)는 Elliptic++ 주소 피처 55개를 "주소의 첫 등장 시점까지의 정보"로 다시 계산해
공개했다. 같은 주소, 같은 시간 분할(첫 등장 time step 1~34 학습 / 35~49 평가)에서 피처만 바꾼다.
  released : Elliptic++가 배포한 피처. 주소의 전체 이력을 요약한 값이라 미래 정보를 포함한다
  resistant: Liang et al.이 재계산한 피처

실행: python x4_address_leakage.py   (data/ellipticpp, data/ellipticpp_lr 필요)
결과: results/x4_address_leakage.csv
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import BASE, DARK_MARKET_STEP, clear_results, evaluate, log_result
from x1_feature_scope import make_model, period_metrics
from x2_address_overlap import EPP_DIR

warnings.filterwarnings("ignore")

EXPERIMENT = "X4"
SEEDS = [0, 1, 2]
LR_PATH = BASE / "data" / "ellipticpp_lr" / "elliptic_pp_leakage_resistant.parquet"


def load():
    """(released, resistant, 피처 열). 두 표는 같은 주소·같은 순서이고 y, split, first_step을 가진다."""
    wallets = pd.read_csv(EPP_DIR / "wallets_features_classes_combined.csv")
    first_step = wallets.groupby("address")["Time step"].min().rename("first_step")
    wallets = wallets.drop_duplicates("address").set_index("address")

    resistant = pd.read_parquet(LR_PATH).set_index("address")
    cols = [c for c in resistant.columns if c != "split"]
    meta = pd.DataFrame({"y": (wallets.loc[resistant.index, "class"] == 1).astype(int),
                         "split": resistant["split"]}).join(first_step)
    released = wallets.loc[resistant.index, cols].join(meta)
    resistant = resistant[cols].join(meta)
    return released, resistant, cols


def models(seed):
    yield "LightGBM", make_model(seed)
    if seed == SEEDS[0]:
        yield "LogReg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def main():
    released, resistant, cols = load()
    print(f"주소 {len(released):,}개, 피처 {len(cols)}개, "
          f"학습 불법 비율 {released.loc[released['split'] == 'train', 'y'].mean():.3f}, "
          f"테스트 불법 비율 {released.loc[released['split'] == 'test', 'y'].mean():.3f}", flush=True)

    clear_results(EXPERIMENT)
    per_run = []
    for name, frame in [("released", released), ("resistant", resistant)]:
        train, test = frame[frame["split"] == "train"], frame[frame["split"] == "test"]
        y, step = test["y"].values, test["first_step"].values
        periods = {"35_49": np.ones(len(test), bool),
                   "35_42": step < DARK_MARKET_STEP, "43_49": step >= DARK_MARKET_STEP}
        for seed in SEEDS:
            for model_name, model in models(seed):
                prob = model.fit(train[cols], train["y"]).predict_proba(test[cols])[:, 1]
                log_result(EXPERIMENT, model_name, f"address:{name}", seed, evaluate(y, prob, 0.5), "threshold=0.5")
                per_run.append({"features": name, "model": model_name,
                                **{f"{k}_{p}": v for p, mask in periods.items()
                                   for k, v in period_metrics(y[mask], prob[mask]).items()}})

    out = pd.DataFrame(per_run).groupby(["features", "model"], sort=False).mean().round(4).reset_index()
    out.to_csv(BASE / "results" / "x4_address_leakage.csv", index=False, encoding="utf-8-sig")
    show = ["features", "model", "f1_35_49", "pr_auc_35_49", "roc_auc_35_49", "f1_35_42", "pr_auc_35_42",
            "f1_43_49", "roc_auc_43_49", "best_f1_43_49"]
    print(out[show].to_string(index=False))


if __name__ == "__main__":
    main()
