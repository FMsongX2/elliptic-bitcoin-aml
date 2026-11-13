"""E0: Elliptic 데이터셋 개요 시각화.

실행: python eda_plots.py  (먼저 download_elliptic.py 로 데이터를 받아 둘 것)
결과: figures/ 아래에 PNG 저장
"""
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data" / "elliptic"
FIG_DIR = BASE / "figures"
SEED = 42

# 클래스별 색: 합법=파랑, 불법=주황, 라벨 없음=중립 회색
COLORS = {"licit": "#2a78d6", "illicit": "#eb6834", "unknown": "#c3c2b7"}
LABELS = {"licit": "합법 (2)", "illicit": "불법 (1)", "unknown": "unknown"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
DARK_MARKET_STEP = 43

plt.rcParams.update({
    "font.family": "Malgun Gothic",
    "axes.unicode_minus": False,
    "figure.dpi": 110,
    "savefig.dpi": 130,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": MUTED,
    "axes.titlecolor": TEXT,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "legend.frameon": False,
})


def load():
    features = pd.read_csv(DATA_DIR / "elliptic_txs_features.csv", header=None)
    features.columns = (["txId", "time_step"]
                        + [f"local_{i}" for i in range(93)]
                        + [f"agg_{i}" for i in range(72)])
    classes = pd.read_csv(DATA_DIR / "elliptic_txs_classes.csv")
    classes["label"] = classes["class"].map({"1": "illicit", "2": "licit", "unknown": "unknown"})
    edges = pd.read_csv(DATA_DIR / "elliptic_txs_edgelist.csv")
    df = features.merge(classes[["txId", "label"]], on="txId")
    return df, edges


def plot_time_steps(df):
    counts = df.groupby(["time_step", "label"]).size().unstack(fill_value=0)
    steps = counts.index.values
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2], "hspace": 0.12})

    bottom = np.zeros(len(steps))
    for lab in ["illicit", "licit", "unknown"]:
        ax1.bar(steps, counts[lab], bottom=bottom, width=0.8, color=COLORS[lab],
                label=LABELS[lab], edgecolor="white", linewidth=0.6)
        bottom += counts[lab].values
    ax1.set_ylabel("거래 수")
    ax1.set_title("time step별 거래 수와 라벨 구성")
    ax1.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=3)

    labeled = counts["illicit"] + counts["licit"]
    ratio = counts["illicit"] / labeled * 100
    ax2.plot(steps, ratio, color=COLORS["illicit"], linewidth=2, marker="o", markersize=4)
    ax2.set_ylabel("불법 비율 (%)\n라벨 있는 거래 중")
    ax2.set_xlabel("time step (약 2주 간격)")
    ax2.set_title("불법 거래 비율 추이")

    for ax in (ax1, ax2):
        ax.axvspan(34.5, 49.5, color=GRID, alpha=0.5, zorder=0)
        ax.axvline(DARK_MARKET_STEP - 0.5, color=MUTED, linestyle="--", linewidth=1)
    ax1.text(35, ax1.get_ylim()[1] * 0.93, "테스트 구간 (35~49)", color=MUTED, fontsize=9)
    ax2.text(DARK_MARKET_STEP - 0.2, ratio.max() * 0.92, "다크마켓 폐쇄 (43~)", color=MUTED, fontsize=9)
    ax2.set_xlim(0.3, 49.7)

    fig.savefig(FIG_DIR / "01_time_steps.png")
    plt.close(fig)


def plot_feature_distributions(df):
    labeled = df[df["label"] != "unknown"]
    local_cols = [c for c in df.columns if c.startswith("local_")]
    # 클래스 간 평균 차이(표준화)가 큰 로컬 피처 6개
    g = labeled.groupby("label")[local_cols]
    smd = ((g.mean().loc["illicit"] - g.mean().loc["licit"]) / labeled[local_cols].std()).abs()
    top = smd.sort_values(ascending=False).index[:6]

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for ax, col in zip(axes.flat, top):
        lo, hi = labeled[col].quantile([0.01, 0.99])
        bins = np.linspace(lo, hi, 50)
        for lab in ["licit", "illicit"]:
            ax.hist(labeled.loc[labeled["label"] == lab, col].clip(lo, hi), bins=bins,
                    density=True, histtype="step", linewidth=2, color=COLORS[lab], label=LABELS[lab])
        ax.set_title(f"{col}  (|SMD| = {smd[col]:.2f})", fontsize=10)
        ax.set_yticks([])
        ax.grid(False)
    axes.flat[0].legend(loc="upper right")
    fig.suptitle("클래스 차이가 큰 로컬 피처 6개의 분포 (밀도, 상하위 1% 절단)",
                 x=0.01, ha="left", fontweight="bold", color=TEXT)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_feature_distributions.png")
    plt.close(fig)


def plot_pca(df):
    labeled = df[df["label"] != "unknown"]
    feat_cols = [c for c in df.columns if c.startswith(("local_", "agg_"))]
    X = StandardScaler().fit_transform(labeled[feat_cols])
    pcs = PCA(n_components=2, random_state=SEED).fit(X)
    Z = pcs.transform(X)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    rng = np.random.default_rng(SEED)
    for lab in ["licit", "illicit"]:
        idx = np.where(labeled["label"].values == lab)[0]
        if lab == "licit":
            idx = rng.choice(idx, size=10000, replace=False)
        ax.scatter(Z[idx, 0], Z[idx, 1], s=6, alpha=0.45, color=COLORS[lab],
                   label=f"{LABELS[lab]}  n={len(idx):,}", linewidths=0)
    ev = pcs.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)")
    ax.set_xlim(np.percentile(Z[:, 0], [0.5, 99.5]))
    ax.set_ylim(np.percentile(Z[:, 1], [0.5, 99.5]))
    ax.set_title("라벨 있는 거래의 PCA 2차원 투영 (피처 165개)")
    ax.legend(loc="upper right", markerscale=3)
    fig.savefig(FIG_DIR / "03_pca.png")
    plt.close(fig)


def plot_degrees(df, edges):
    in_deg = edges["txId2"].value_counts().reindex(df["txId"], fill_value=0)
    out_deg = edges["txId1"].value_counts().reindex(df["txId"], fill_value=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, deg, name in [(axes[0], in_deg, "in-degree (받은 거래)"),
                          (axes[1], out_deg, "out-degree (보낸 거래)")]:
        vc = deg.value_counts().sort_index()
        vc = vc[vc.index > 0]
        ax.scatter(vc.index, vc.values, s=14, color=COLORS["licit"], linewidths=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(name)
        ax.set_ylabel("노드 수")
        zero = (deg == 0).mean() * 100
        ax.set_title(f"{name} 분포 (log-log) · 0인 노드 {zero:.1f}%", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_degree_distribution.png")
    plt.close(fig)


def plot_subgraph(df, edges):
    G = nx.from_pandas_edgelist(edges, "txId1", "txId2", create_using=nx.DiGraph)
    label_of = df.set_index("txId")["label"]

    # 불법 노드가 많이 섞인 2-hop 이웃 그래프 하나를 고른다
    rng = np.random.default_rng(SEED)
    illicit_ids = df.loc[df["label"] == "illicit", "txId"].values
    best = None
    for seed_node in rng.choice(illicit_ids, size=300, replace=False):
        nodes = nx.ego_graph(G, seed_node, radius=2, undirected=True).nodes
        if not 40 <= len(nodes) <= 150:
            continue
        n_illicit = sum(label_of[n] == "illicit" for n in nodes)
        n_licit = sum(label_of[n] == "licit" for n in nodes)
        score = min(n_illicit, 10) + min(n_licit, 10)
        if best is None or score > best[0]:
            best = (score, seed_node, nodes)
    _, center, nodes = best
    H = G.subgraph(nodes)

    pos = nx.spring_layout(H, seed=SEED, k=0.35)
    fig, ax = plt.subplots(figsize=(9, 8))
    nx.draw_networkx_edges(H, pos, ax=ax, edge_color="#a8a7a1", width=0.8, arrowsize=7, alpha=0.8)
    for lab in ["unknown", "licit", "illicit"]:
        nl = [n for n in H.nodes if label_of[n] == lab]
        nx.draw_networkx_nodes(H, pos, nodelist=nl, ax=ax, node_size=70, node_color=COLORS[lab],
                               edgecolors="white", linewidths=1, label=f"{LABELS[lab]} ({len(nl)})")
    nx.draw_networkx_nodes(H, pos, nodelist=[center], ax=ax, node_size=220, node_color=COLORS["illicit"],
                           edgecolors=TEXT, linewidths=1.5)
    step = int(df.loc[df["txId"] == center, "time_step"].iloc[0])
    ax.set_title(f"불법 거래 1건(테두리) 주변 2-hop 거래 그래프 · time step {step}", loc="left")
    ax.legend(loc="lower left", scatterpoints=1)
    ax.axis("off")
    fig.savefig(FIG_DIR / "05_subgraph.png")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(exist_ok=True)
    df, edges = load()
    plot_time_steps(df)
    plot_feature_distributions(df)
    plot_pca(df)
    plot_degrees(df, edges)
    plot_subgraph(df, edges)
    print(f"저장 완료: {FIG_DIR}")


if __name__ == "__main__":
    main()
