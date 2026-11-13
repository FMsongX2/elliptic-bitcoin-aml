"""X2: 학습·테스트 구간 사이의 주소 중복과 탐지 성능의 관계.

Elliptic의 라벨은 입력 주소를 소유한 실체에서 온다(Weber et al., 2019). 같은 주소가 학습 구간과
테스트 구간에 모두 나타나면 모델은 거래 패턴이 아니라 이미 본 실체를 다시 알아보는 것일 수 있다
(Šafář et al., 2026). Elliptic++의 주소-거래 엣지로 이를 직접 센다.

테스트 거래(35~49)를 입력 주소 기준으로 세 그룹으로 나눈다.
  seen_illicit : 입력 주소 중 하나가 학습 구간(1~34) 불법 거래의 입력 주소였다
  seen_other   : 입력 주소 중 하나가 학습 구간 거래에 (입력 또는 출력으로) 등장했다
  unseen       : 입력 주소가 학습 구간에 한 번도 등장하지 않았다

그룹별로 LightGBM(E2 설정, seed 0)의 불법 Recall과 합법 오탐률을 기간별로 기록한다.

실행: python x2_address_overlap.py   (data/ellipticpp 에 Elliptic++ 필요)
결과: results/x2_address_overlap.csv
"""
import warnings

import numpy as np
import pandas as pd

from common import BASE, DARK_MARKET_STEP, TEST_STEPS, load_dataset, split
from x1_feature_scope import ADDR, NBR_IN, NBR_OUT, TX, make_model

warnings.filterwarnings("ignore", category=UserWarning)

EPP_DIR = BASE / "data" / "ellipticpp" / "Elliptic++ Dataset"
MODELS = {"all": TX + ADDR + NBR_IN + NBR_OUT, "tx+nbr_in": TX + NBR_IN}
GROUPS = ["seen_illicit", "seen_other", "unseen", "no_address"]


def address_groups(df_all):
    """txId → 그룹. df_all은 unknown을 포함한 전체 거래(time_step, y)."""
    addr_tx = pd.read_csv(EPP_DIR / "AddrTx_edgelist.csv").rename(columns={"input_address": "address"})
    tx_addr = pd.read_csv(EPP_DIR / "TxAddr_edgelist.csv").rename(columns={"output_address": "address"})
    meta = df_all.set_index("txId")[["time_step", "y"]]
    addr_tx = addr_tx.join(meta, on="txId")
    tx_addr = tx_addr.join(meta, on="txId")

    train_max = min(TEST_STEPS) - 1
    seen_any = set(addr_tx.loc[addr_tx["time_step"] <= train_max, "address"]) \
        | set(tx_addr.loc[tx_addr["time_step"] <= train_max, "address"])
    seen_illicit = set(addr_tx.loc[(addr_tx["time_step"] <= train_max) & (addr_tx["y"] == 1), "address"])

    test_in = addr_tx[addr_tx["time_step"] > train_max]
    flags = pd.DataFrame({
        "txId": test_in["txId"].values,
        "hit_illicit": test_in["address"].isin(seen_illicit).values,
        "hit_any": test_in["address"].isin(seen_any).values,
    }).groupby("txId").any()
    group = np.where(flags["hit_illicit"], "seen_illicit", np.where(flags["hit_any"], "seen_other", "unseen"))

    n_test_addr = test_in["address"].nunique()
    print(f"테스트 구간 입력 주소 {n_test_addr:,}개 중 학습 구간에 등장한 주소 "
          f"{test_in.loc[test_in['address'].isin(seen_any), 'address'].nunique():,}개")
    return pd.Series(group, index=flags.index)


def main():
    df_all = load_dataset(labeled_only=False)
    df = df_all[df_all["y"].notna()].copy()
    df["y"] = df["y"].astype(int)
    train, val, test = split(df)
    trainval = pd.concat([train, val])

    test = test.copy()
    test["group"] = test["txId"].map(address_groups(df_all)).fillna("no_address")
    test["period"] = np.where(test["time_step"] < DARK_MARKET_STEP, "35_42", "43_49")

    rows = []
    for name, cols in MODELS.items():
        prob = make_model(0).fit(trainval[cols], trainval["y"]).predict_proba(test[cols])[:, 1]
        test["pred"] = prob >= 0.5
        for period in ["35_42", "43_49"]:
            for group in GROUPS:
                part = test[(test["period"] == period) & (test["group"] == group)]
                illicit, licit = part[part["y"] == 1], part[part["y"] == 0]
                rows.append({
                    "model": name, "period": period, "group": group,
                    "n_illicit": len(illicit), "n_licit": len(licit),
                    "illicit_recall": illicit["pred"].mean() if len(illicit) else np.nan,
                    "licit_fpr": licit["pred"].mean() if len(licit) else np.nan,
                })
    out = pd.DataFrame(rows).round(4)
    out.to_csv(BASE / "results" / "x2_address_overlap.csv", index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
