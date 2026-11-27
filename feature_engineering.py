"""설정 B(거래 시점 정보만)에서 쓰는 가공 피처.

모든 피처는 거래가 일어난 순간에 알 수 있는 정보(거래 자체 TX, 선행 거래 집계 NBR_IN,
선행 거래 쪽 연결)만으로 만든다. 라벨은 쓰지 않는다.

    from feature_engineering import build_engineered_features
    fe = build_engineered_features(load_dataset(labeled_only=False), load_edges())

피처 묶음
- shape  (`fe_`)  : 입출력 개수를 실제 개수로 되돌린 값, 거래 모양 플래그(입력1·출력1 등), 비율
- pred   (`fp_`)  : 선행 거래들의 모양 (선행 거래 중 입력1·출력1 비율 등)
- pct    (`pct_`) : 직전 time step 분포 기준 백분위. 수수료·금액처럼 시기마다 수준이 바뀌는
                    값을 "그 시기 안에서 상대적으로 큰가"로 바꿔 시간 변화의 영향을 줄인다.
                    기준 분포는 직전 time step(과거)만 쓴다

시점을 담은 피처
- HEIGHT_COLS(`agg_6~11`, 선행 거래의 블록 높이)는 시간이 지날수록 커지는 값이라 사실상 날짜다.
  모델이 "언제의 거래인지"를 외우는 데 쓸 수 있어 변형에서 뺄 수 있게 따로 둔다.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from common import NBR_IN, TX, z_to_count

HEIGHT_COLS = [f"agg_{i}" for i in range(6, 12)]
SHAPE_PREFIX = "fe_"
PRED_PREFIX = "fp_"
PCT_PREFIX = "pct_"


def _shape(df_all):
    n_in = z_to_count(df_all["local_3"].values)
    n_out = z_to_count(df_all["local_4"].values)
    n_uin = z_to_count(df_all["local_5"].values)
    n_uout = z_to_count(df_all["local_13"].values)
    return pd.DataFrame({
        "fe_n_in": n_in,
        "fe_n_out": n_out,
        "fe_n_uniq_in": n_uin,
        "fe_n_uniq_out": n_uout,
        "fe_1in1out": ((n_in == 1) & (n_out == 1)).astype(int),
        "fe_1in2out": ((n_in == 1) & (n_out == 2)).astype(int),     # 송금 + 잔돈 (peel 모양)
        "fe_many_in_1out": ((n_in >= 3) & (n_out == 1)).astype(int),  # 여러 곳에서 모아 한 곳으로
        "fe_fanout": (n_out >= 3).astype(int),                       # 여러 곳으로 나눠 보냄
        "fe_out_in_ratio": n_out / n_in,
        "fe_uniq_in_ratio": n_uin / n_in,     # 1보다 작으면 같은 주소에서 여러 입력
        "fe_uniq_out_ratio": n_uout / n_out,  # 1보다 작으면 같은 주소로 여러 출력
    })


def _pred_shape(df_all, edges, shape):
    n = len(df_all)
    pos = pd.Series(np.arange(n), index=df_all["txId"].values)
    src = pos.loc[edges["txId1"]].values
    dst = pos.loc[edges["txId2"]].values
    P = sp.csr_matrix((np.ones(len(src)), (dst, src)), shape=(n, n))  # P[i, j] = 1 이면 j가 i의 선행 거래
    deg = np.asarray(P.sum(axis=1)).ravel()

    def mean_over_preds(v):
        return np.divide(P @ v, deg, out=np.full(n, np.nan), where=deg > 0)

    return pd.DataFrame({
        "fp_n_pred": deg,
        "fp_pred_1in1out": mean_over_preds(shape["fe_1in1out"].values.astype(float)),
        "fp_pred_1in2out": mean_over_preds(shape["fe_1in2out"].values.astype(float)),
        "fp_pred_fanout": mean_over_preds(shape["fe_fanout"].values.astype(float)),
        "fp_pred_n_out": mean_over_preds(shape["fe_n_out"].values.astype(float)),
    })


def _prev_step_percentile(df_all, cols):
    """각 값이 직전 time step의 분포에서 몇 % 위치인지. 첫 time step은 자기 분포를 쓴다."""
    step = df_all["time_step"].values
    out = {}
    for c in cols:
        x = df_all[c].values
        pct = np.empty(len(x))
        for t in np.unique(step):
            ref = np.sort(x[step == (t - 1 if t > 1 else t)])
            idx = step == t
            pct[idx] = np.searchsorted(ref, x[idx], side="right") / len(ref)
        out[f"{PCT_PREFIX}{c}"] = pct
    return pd.DataFrame(out)


def build_engineered_features(df_all, edges):
    """df_all: 전체 거래(unknown 포함). 반환: txId + fe_/fp_/pct_ 피처."""
    shape = _shape(df_all)
    pred = _pred_shape(df_all, edges, shape)
    pct_cols = TX + [c for c in NBR_IN if c not in HEIGHT_COLS]
    pct = _prev_step_percentile(df_all, pct_cols)
    out = pd.concat([shape, pred, pct], axis=1)
    out.insert(0, "txId", df_all["txId"].values)
    return out
