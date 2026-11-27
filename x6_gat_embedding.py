"""X6: GAT, 그리고 GNN 임베딩 + LightGBM.

X3과 같은 인코더 구조(2층, 은닉 128)와 학습 절차(1~34 서브그래프, 300 epoch, 코사인 감쇠)에서
  1) 인코더 종류를 바꾼다: mlp(연결 정보 없음), sage(GraphSAGE), gat(GAT, 머리 4개)
  2) 인코더의 128차원 임베딩을 LightGBM(E2 설정)에 넣는다
       head       : 인코더에 붙은 선형 분류기 (X3과 같은 방식)
       emb        : 임베딩만
       raw + emb  : 원 피처 + 임베딩
     기준은 같은 시드의 원 피처 LightGBM(raw)이다.

mlp 임베딩은 대조군이다. "raw + sage 임베딩"이 "raw + mlp 임베딩"보다 높아야 그 차이를
그래프의 기여로 볼 수 있다.

인코더와 LightGBM은 같은 1~34 라벨로 학습한다. 학습 노드의 임베딩은 그 라벨에 맞춰진 값이라
테스트 노드의 임베딩과 분포가 다를 수 있다(스태킹 누수). 문헌의 관행을 따른 것이고 교차 적합은 하지 않았다.

피처셋: tx(거래 자체 21개), time(설정 B, 57개), all(설정 A, 165개)

실행: python x6_gat_embedding.py   (torch, torch_geometric 필요: requirements-gnn.txt)
결과: results/x6_gat_embedding.csv
"""
import warnings

import numpy as np
import pandas as pd
import torch

from common import (BASE, DARK_MARKET_STEP, FEATURE_SETS, TEST_STEPS, TRAIN_STEPS, TX, VAL_STEPS,
                    clear_results, evaluate, load_dataset, load_edges, log_result)
from x1_feature_scope import make_model, period_metrics
from x3_gnn import CLIP, DEVICE, Net, build_edge_index, fit, part, predict

warnings.filterwarnings("ignore", category=UserWarning)

EXPERIMENT = "X6"
SEEDS = list(range(10))
ENCODERS = ["mlp", "sage", "gat"]
VARIANTS = {"tx": TX, "time": FEATURE_SETS["time"], "all": FEATURE_SETS["all"]}
FIT_STEPS = [*TRAIN_STEPS, *VAL_STEPS]


@torch.no_grad()
def embed(model, data):
    """라벨 있는 노드의 128차원 임베딩."""
    x, _, edge_index, labeled = data
    model.eval()
    return model.embed(x, edge_index)[labeled].cpu().numpy()


def lightgbm_prob(seed, train_x, train_y, test_x):
    return make_model(seed).fit(train_x, train_y).predict_proba(test_x)[:, 1]


def main():
    df = load_dataset(labeled_only=False)
    step = torch.from_numpy(df["time_step"].values).to(DEVICE)
    y = torch.from_numpy(df["y"].values.astype(np.float32)).to(DEVICE)
    edge_index = build_edge_index(df, load_edges())

    labeled = df["y"].notna().values
    fit_rows = labeled & df["time_step"].isin(FIT_STEPS).values
    test_rows = labeled & df["time_step"].isin(TEST_STEPS).values
    fit_y, test_y = df.loc[fit_rows, "y"].values.astype(int), df.loc[test_rows, "y"].values.astype(int)
    test_step = df.loc[test_rows, "time_step"].values
    periods = {"35_49": np.ones(len(test_y), bool),
               "35_42": test_step < DARK_MARKET_STEP,
               "43_49": test_step >= DARK_MARKET_STEP}
    print(f"device={DEVICE}, 학습 {fit_rows.sum():,}건, 테스트 {test_rows.sum():,}건", flush=True)

    clear_results(EXPERIMENT)
    runs = []

    def record(variant, encoder, inputs, seed, prob):
        log_result(EXPERIMENT, f"{encoder}:{inputs}", variant, seed, evaluate(test_y, prob, 0.5), "threshold=0.5")
        runs.append({"variant": variant, "encoder": encoder, "inputs": inputs,
                     **{f"{k}_{p}": v for p, mask in periods.items()
                        for k, v in period_metrics(test_y[mask], prob[mask]).items()}})

    for variant, cols in VARIANTS.items():
        raw_fit, raw_test = df.loc[fit_rows, cols].values, df.loc[test_rows, cols].values
        for seed in SEEDS:
            record(variant, "none", "raw", seed, lightgbm_prob(seed, raw_fit, fit_y, raw_test))

        x = torch.from_numpy(df[cols].values.astype(np.float32)).clamp(-CLIP, CLIP).to(DEVICE)
        fit_part, test_part = part(x, y, edge_index, step, FIT_STEPS), part(x, y, edge_index, step, TEST_STEPS)
        for encoder in ENCODERS:
            for seed in SEEDS:
                torch.manual_seed(seed)
                model = fit(Net(len(cols), encoder).to(DEVICE), fit_part)
                record(variant, encoder, "head", seed, predict(model, test_part[0], test_part[2], test_part[3]))
                emb_fit, emb_test = embed(model, fit_part), embed(model, test_part)
                record(variant, encoder, "emb", seed, lightgbm_prob(seed, emb_fit, fit_y, emb_test))
                record(variant, encoder, "raw+emb", seed,
                       lightgbm_prob(seed, np.hstack([raw_fit, emb_fit]), fit_y, np.hstack([raw_test, emb_test])))

        done = pd.DataFrame(runs)
        done = done[done["variant"] == variant].groupby(["encoder", "inputs"], sort=False)
        for (encoder, inputs), g in done:
            print(f"{variant:5s} {encoder:5s} {inputs:8s} PR-AUC {g['pr_auc_35_49'].mean():.3f}±{g['pr_auc_35_49'].std():.3f}"
                  f"  F1 {g['f1_35_49'].mean():.3f}±{g['f1_35_49'].std():.3f}"
                  f" | 35~42 PR-AUC {g['pr_auc_35_42'].mean():.3f} F1 {g['f1_35_42'].mean():.3f}"
                  f" | 43~49 ROC-AUC {g['roc_auc_43_49'].mean():.3f} bestF1 {g['best_f1_43_49'].mean():.3f}", flush=True)

    grouped = pd.DataFrame(runs).groupby(["variant", "encoder", "inputs"], sort=False)
    summary = grouped.mean().round(4)
    summary["pr_auc_std"] = grouped["pr_auc_35_49"].std().round(4)
    summary["f1_std"] = grouped["f1_35_49"].std().round(4)
    out = BASE / "results" / "x6_gat_embedding.csv"
    summary.reset_index().to_csv(out, index=False, encoding="utf-8-sig")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
