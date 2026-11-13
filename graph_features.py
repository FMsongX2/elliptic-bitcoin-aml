"""E3: 엣지리스트로 직접 계산하는 그래프 피처.

라벨은 전혀 쓰지 않는다(누수 없음). 엣지는 같은 time step 안에서만 이어지므로
모든 피처는 사실상 time step별 그래프 안에서 계산된다.

    from graph_features import build_graph_features
    gf = build_graph_features(load_dataset(labeled_only=False), load_edges())  # txId + g_*/nin_*/nout_*/n2_* 열

피처 그룹
- 구조(struct): 차수, PageRank, 이웃 차수, 2-hop 크기, peel chain 길이
  (연결 컴포넌트 크기는 넣지 않는다: time step 하나가 정확히 컴포넌트 하나라서
   그 값은 time step의 거래 수와 같고, 시점 정보를 몰래 알려 주는 피처가 된다)
- 방향별 이웃 평균(nbr): 돈을 보낸 쪽(in) / 받은 쪽(out) 거래의 로컬 피처 평균
- 2-hop 평균(nbr2): 이웃의 이웃까지 포함한 로컬 피처 평균 (기존 집계 피처는 1-hop뿐)
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp



def _row_normalize(M):
    deg = np.asarray(M.sum(axis=1)).ravel()
    inv = np.divide(1.0, deg, out=np.zeros_like(deg, dtype=float), where=deg > 0)
    return sp.diags(inv) @ M


def _pagerank(A, alpha=0.85, tol=1e-10, max_iter=200):
    """방향 그래프 PageRank (A[i, j] = 1 이면 i -> j). 평균이 1이 되도록 n을 곱해 반환."""
    n = A.shape[0]
    out_deg = np.asarray(A.sum(axis=1)).ravel()
    P = _row_normalize(A).T.tocsr()
    dangling = out_deg == 0
    r = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        r_new = alpha * (P @ r + r[dangling].sum() / n) + (1 - alpha) / n
        if np.abs(r_new - r).sum() < tol:
            r = r_new
            break
        r = r_new
    return r * n


def _peel_chain_length(A, in_deg, out_deg):
    """1입력 -> 1~2출력 거래(peel 패턴)가 뒤에서부터 몇 번 연속됐는지."""
    is_peel = (in_deg == 1) & (out_deg >= 1) & (out_deg <= 2)
    pred = np.full(A.shape[0], -1)
    AT = A.T.tocsr()
    single = np.flatnonzero(in_deg == 1)
    pred[single] = AT.indices[AT.indptr[single]]

    # 엣지 그래프는 DAG라서 (가장 긴 체인 길이)번 안에 반드시 수렴한다
    chain = is_peel.astype(int)
    has_pred = pred >= 0
    for _ in range(A.shape[0]):
        prev = np.zeros_like(chain)
        prev[has_pred] = chain[pred[has_pred]]
        new = np.where(is_peel, 1 + prev, 0)
        if np.array_equal(new, chain):
            break
        chain = new
    return chain


def build_graph_features(df_all, edges):
    """df_all: 전체 거래(unknown 포함, txId/local_* 열 필요). 반환: txId + 그래프 피처."""
    n = len(df_all)
    pos = pd.Series(np.arange(n), index=df_all["txId"].values)
    src = pos.loc[edges["txId1"]].values
    dst = pos.loc[edges["txId2"]].values

    A = sp.csr_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
    U = ((A + A.T) > 0).astype(float).tocsr()

    in_deg = np.asarray(A.sum(axis=0)).ravel()
    out_deg = np.asarray(A.sum(axis=1)).ravel()
    deg = np.asarray(U.sum(axis=1)).ravel()

    nb_deg_max = U.multiply(deg[None, :]).max(axis=1).toarray().ravel()
    nb_deg_mean = np.divide(U @ deg, deg, out=np.zeros(n), where=deg > 0)

    U2 = ((U @ U + U) > 0).astype(float).tolil()
    U2.setdiag(0)
    U2 = U2.tocsr()
    U2.eliminate_zeros()
    two_hop = np.asarray(U2.sum(axis=1)).ravel()

    feats = {
        "txId": df_all["txId"].values,
        "g_in_deg": in_deg,
        "g_out_deg": out_deg,
        "g_deg": deg,
        "g_out_in_ratio": (out_deg + 1) / (in_deg + 1),
        "g_pagerank": _pagerank(A),
        "g_nb_deg_max": nb_deg_max,
        "g_nb_deg_mean": nb_deg_mean,
        "g_2hop_size": two_hop,
        "g_peel_chain": _peel_chain_length(A, in_deg, out_deg),
    }

    local_cols = [c for c in df_all.columns if c.startswith("local_")]
    X = df_all[local_cols].to_numpy(dtype=np.float32)
    blocks = {
        "nin": _row_normalize(A.T.tocsr()) @ X,   # 돈을 보낸 거래(선행)들의 평균
        "nout": _row_normalize(A) @ X,            # 돈을 받은 거래(후행)들의 평균
        "n2": _row_normalize(U2) @ X,             # 2-hop 이웃 평균
    }
    for prefix, M in blocks.items():
        for j, c in enumerate(local_cols):
            feats[f"{prefix}_{c}"] = M[:, j].astype(np.float32)
    return pd.DataFrame(feats)


STRUCT_PREFIX = "g_"
NBR_PREFIXES = ("nin_", "nout_")
NBR2_PREFIX = "n2_"
