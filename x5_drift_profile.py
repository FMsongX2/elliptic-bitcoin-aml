"""X5: 다크마켓 폐쇄(43번째 time step) 전후로 불법 거래의 피처가 어떻게 달라졌는가.

1) E1 깊이 2 트리 규칙(local_52 ≤ -0.488 이고 local_13 ≤ -0.053)을 만족하는 비율을 기간·클래스별로 센다.
   local_52 = 입력 주소 수명의 최댓값, local_13 = 고유 출력 주소 수 (Šafář et al., 2026).
2) 불법 거래의 피처 중앙값이 35~42와 43~49 사이에서 가장 크게 움직인 피처를 고른다.
   이동 폭은 라벨 있는 거래 전체의 표준편차로 나눈 값이다.

실행: python x5_drift_profile.py   (feature_semantics.csv 필요: build_feature_semantics.py)
결과: results/x5_rule_by_period.csv, results/x5_feature_shift.csv
"""
import pandas as pd

from common import BASE, DARK_MARKET_STEP, TEST_STEPS, feature_cols, load_dataset

TOP = 15
RULE = {"local_52": -0.488, "local_13": -0.053}


def main():
    df = load_dataset(labeled_only=True)
    semantics = pd.read_csv(BASE / "feature_semantics.csv").set_index("column")["description"]
    edges = [0, min(TEST_STEPS) - 1, DARK_MARKET_STEP - 1, max(TEST_STEPS)]
    df["period"] = pd.cut(df["time_step"], edges, labels=["01_34", "35_42", "43_49"])
    df["class"] = df["y"].map({1: "illicit", 0: "licit"})

    hit = pd.DataFrame({col: df[col] <= cut for col, cut in RULE.items()})
    hit["rule"] = hit.all(axis=1)
    rule = (hit.join(df[["period", "class"]])
            .groupby(["period", "class"], observed=True)
            .agg(n=("rule", "size"), **{f"{c}_le_cut": (c, "mean") for c in RULE}, rule=("rule", "mean"))
            .round(4).reset_index())
    rule.to_csv(BASE / "results" / "x5_rule_by_period.csv", index=False, encoding="utf-8-sig")
    print(rule.to_string(index=False))

    cols = feature_cols(df, "all")
    median = df.groupby(["class", "period"], observed=True)[cols].median()
    shift = ((median.loc[("illicit", "43_49")] - median.loc[("illicit", "35_42")]) / df[cols].std()).abs()
    top = shift.sort_values(ascending=False).head(TOP).index
    table = pd.DataFrame({
        "shift_in_std": shift[top],
        "illicit_01_34": median.loc[("illicit", "01_34"), top],
        "illicit_35_42": median.loc[("illicit", "35_42"), top],
        "illicit_43_49": median.loc[("illicit", "43_49"), top],
        "licit_43_49": median.loc[("licit", "43_49"), top],
        "description": semantics.reindex(top),
    }).round(3).rename_axis("column").reset_index()
    table.to_csv(BASE / "results" / "x5_feature_shift.csv", index=False, encoding="utf-8-sig")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
