"""X3: 그래프의 순수 기여 측정 (GraphSAGE 대 같은 피처의 MLP 대 엣지를 섞은 그래프).

같은 피처, 같은 깊이·폭, 같은 학습 절차에서 연결 정보만 바꾼다.
  mlp      : 연결 정보 없음
  sage     : 실제 거래 그래프 (방향을 무시한 양방향 엣지)
  shuffled : time step 안에서 엣지의 도착 노드만 섞은 그래프. 엣지 수와 각 노드의 차수는 유지된다

피처셋은 X1의 구분을 따른다. 이웃 정보가 피처에 들어 있지 않은 `tx`에서 GNN이 올리는 폭이
그래프 기여의 가장 깨끗한 측정이다.

학습 절차는 팀 공통 규칙과 같다.
  1) 1~30 서브그래프로 학습한 모델로 31~34에서 F1 최대 임계값을 고른다
  2) 같은 설정으로 1~34 서브그래프를 다시 학습해 35~49를 평가한다
time step 사이에는 엣지가 없으므로 학습 중 메시지 전달과 손실은 테스트 구간 노드를 보지 않는다.

epoch 수는 고정한다. 검증 구간(31~34)의 PR-AUC는 테스트 성능과 거의 따로 움직여서 조기 종료
기준으로 쓸 수 없었다. 학습률은 코사인 감쇠로 0까지 줄여 마지막 epoch의 진동을 없앤다.

실행: python x3_gnn.py   (torch, torch_geometric 필요: requirements-gnn.txt)
결과: results/x3_gnn.csv
"""
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.nn import GATConv, SAGEConv
from torch_geometric.utils import subgraph

from common import (BASE, DARK_MARKET_STEP, TEST_STEPS, TRAIN_STEPS, VAL_STEPS, clear_results, evaluate,
                    load_dataset, load_edges, log_result, tune_threshold)
from x1_feature_scope import ADDR, NBR_IN, NBR_OUT, TX, period_metrics

EXPERIMENT = "X3"
SEEDS = list(range(10))
VARIANTS = {"tx": TX, "tx+nbr_in": TX + NBR_IN, "all": TX + ADDR + NBR_IN + NBR_OUT}
GRAPHS = ["mlp", "sage", "shuffled"]

HIDDEN = 128
DROPOUT = 0.3
LR = 0.01
WEIGHT_DECAY = 5e-4
EPOCHS = 300
CLIP = 10.0  # z-score 피처의 긴 꼬리를 자른다. 모든 모델에 같게 적용한다
GAT_HEADS = 4  # 머리 4개 × 32차원을 이어 붙여 은닉 128을 맞춘다

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_layer(kind, n_in):
    if kind == "mlp":
        return nn.Linear(n_in, HIDDEN)
    if kind == "sage":
        return SAGEConv(n_in, HIDDEN)
    if kind == "gat":
        return GATConv(n_in, HIDDEN // GAT_HEADS, heads=GAT_HEADS)
    raise ValueError(kind)


class Net(nn.Module):
    """2층 인코더(은닉 128) + 선형 분류기. kind가 연결 정보를 쓰는 방식을 정한다."""

    def __init__(self, n_in, kind):
        super().__init__()
        self.use_graph = kind != "mlp"
        self.l1, self.l2 = make_layer(kind, n_in), make_layer(kind, HIDDEN)
        self.head = nn.Linear(HIDDEN, 1)
        self.drop = nn.Dropout(DROPOUT)

    def embed(self, x, edge_index):
        for layer in (self.l1, self.l2):
            x = layer(x, edge_index) if self.use_graph else layer(x)
            x = self.drop(torch.relu(x))
        return x

    def forward(self, x, edge_index):
        return self.head(self.embed(x, edge_index)).squeeze(-1)


def build_edge_index(df, edges, shuffle_seed=None):
    """양방향 edge_index. shuffle_seed가 있으면 time step 안에서 도착 노드를 섞는다."""
    index = pd.Series(np.arange(len(df)), index=df["txId"].values)
    src, dst = index.loc[edges["txId1"]].values, index.loc[edges["txId2"]].values
    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        step = df["time_step"].values[src]
        dst = dst.copy()
        for t in np.unique(step):
            pos = np.flatnonzero(step == t)
            dst[pos] = rng.permutation(dst[pos])
    both = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])])
    return torch.from_numpy(both).long().to(DEVICE)


def part(x, y, edge_index, step, steps):
    """해당 time step들만 남긴 서브그래프 (x, y, edge_index, 라벨 있는 노드 마스크)."""
    mask = torch.isin(step, torch.tensor(list(steps), device=DEVICE))
    sub_edges, _ = subgraph(mask, edge_index, relabel_nodes=True, num_nodes=len(step))
    return x[mask], y[mask], sub_edges, ~torch.isnan(y[mask])


@torch.no_grad()
def predict(model, x, edge_index, labeled):
    model.eval()
    return torch.sigmoid(model(x, edge_index)[labeled]).cpu().numpy()


def fit(model, train):
    x, y, edge_index, labeled = train
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        loss_fn(model(x, edge_index)[labeled], y[labeled]).backward()
        optimizer.step()
        scheduler.step()
    return model


def run_one(x, y, edge_index, step, kind, seed):
    train = part(x, y, edge_index, step, TRAIN_STEPS)
    val = part(x, y, edge_index, step, VAL_STEPS)
    trainval = part(x, y, edge_index, step, [*TRAIN_STEPS, *VAL_STEPS])
    tx, ty, t_edges, t_labeled = part(x, y, edge_index, step, TEST_STEPS)

    torch.manual_seed(seed)
    model = fit(Net(x.shape[1], kind).to(DEVICE), train)
    threshold = tune_threshold(val[1][val[3]].cpu().numpy(), predict(model, val[0], val[2], val[3]))

    torch.manual_seed(seed)
    model = fit(Net(x.shape[1], kind).to(DEVICE), trainval)
    prob = predict(model, tx, t_edges, t_labeled)
    test_step = step[torch.isin(step, torch.tensor(list(TEST_STEPS), device=DEVICE))][t_labeled].cpu().numpy()
    return prob, ty[t_labeled].cpu().numpy().astype(int), test_step, threshold


def main():
    df = load_dataset(labeled_only=False)
    edges = load_edges()
    step = torch.from_numpy(df["time_step"].values).to(DEVICE)
    y = torch.from_numpy(df["y"].values.astype(np.float32)).to(DEVICE)
    real_edges = build_edge_index(df, edges)
    print(f"device={DEVICE}, nodes={len(df):,}, edges={real_edges.shape[1] // 2:,}", flush=True)

    clear_results(EXPERIMENT)
    rows = []
    for name, cols in VARIANTS.items():
        x = torch.from_numpy(df[cols].values.astype(np.float32)).clamp(-CLIP, CLIP).to(DEVICE)
        for graph in GRAPHS:
            per_seed = []
            for seed in SEEDS:
                edge_index = build_edge_index(df, edges, seed) if graph == "shuffled" else real_edges
                kind = "mlp" if graph == "mlp" else "sage"
                prob, truth, test_step, threshold = run_one(x, y, edge_index, step, kind, seed)
                log_result(EXPERIMENT, graph, name, seed, evaluate(truth, prob, 0.5), "threshold=0.5")
                periods = {"35_49": np.ones(len(truth), bool),
                           "35_42": test_step < DARK_MARKET_STEP,
                           "43_49": test_step >= DARK_MARKET_STEP}
                metrics = {f"{k}_{p}": v for p, mask in periods.items()
                           for k, v in period_metrics(truth[mask], prob[mask]).items()}
                metrics["f1_tuned_35_49"] = evaluate(truth, prob, threshold)["f1"]
                per_seed.append(metrics)
            frame = pd.DataFrame(per_seed)
            mean, std = frame.mean(), frame.std()
            rows.append({"variant": name, "graph": graph, "n_features": len(cols),
                         **mean.round(4).to_dict(),
                         "f1_std": round(std["f1_35_49"], 4), "pr_auc_std": round(std["pr_auc_35_49"], 4)})
            print(f"{name:10s} {graph:8s} F1 {mean['f1_35_49']:.3f}±{std['f1_35_49']:.3f}"
                  f"  tunedF1 {mean['f1_tuned_35_49']:.3f}"
                  f"  PR-AUC {mean['pr_auc_35_49']:.3f}±{std['pr_auc_35_49']:.3f}"
                  f" | 35~42 F1 {mean['f1_35_42']:.3f} PR-AUC {mean['pr_auc_35_42']:.3f}"
                  f" | 43~49 ROC-AUC {mean['roc_auc_43_49']:.3f} bestF1 {mean['best_f1_43_49']:.3f}", flush=True)

    out = BASE / "results" / "x3_gnn.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
