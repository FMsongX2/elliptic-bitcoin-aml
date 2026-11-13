"""Šafář et al. (2026)가 공개한 Elliptic 피처 의미를 받아 우리 열 이름에 맞춘 표를 만든다.

출처: https://github.com/nesfit/DeanonymizedEllipticBitcoinDataset
인용: Šafář, M., Pluskal, J., Veselý, V., Ryšavý, O. (2026). The enemy of reproducibility is opacity:
      What's inside the Elliptic bitcoin dataset (and why it is wrong). FSI: Digital Investigation 57, 302124.

Elliptic 피처 번호 k는 features CSV의 k번째 열(0번 열은 txId)이다.
  1        → time_step
  2 ~ 22   → local_0 ~ local_20   (scope=tx, 거래 자체)
  23 ~ 94  → local_21 ~ local_92  (scope=address, 입출력 주소 누적 통계)
  95 ~ 166 → agg_0 ~ agg_71       (scope=neighbor_tx, 선행·후속 거래 집계)

실행: python build_feature_semantics.py
결과: feature_semantics.csv (column, elliptic_feature_no, scope, description)
"""
import csv
import re
import urllib.request

from common import BASE

SOURCE = "https://raw.githubusercontent.com/nesfit/DeanonymizedEllipticBitcoinDataset/HEAD/README.md"
ROW = re.compile(r"\|\s*Feature\s+(\d+)\s*\|\s*(.*?)\s*\|")


def column_and_scope(no):
    if no == 1:
        return "time_step", "meta"
    if no <= 22:
        return f"local_{no - 2}", "tx"
    if no <= 94:
        return f"local_{no - 2}", "address"
    return f"agg_{no - 95}", "neighbor_tx"


def main():
    text = urllib.request.urlopen(SOURCE).read().decode("utf-8")
    features = sorted({(int(no), desc) for no, desc in ROW.findall(text)})
    assert [no for no, _ in features] == list(range(1, 167)), "피처 번호 1~166이 모두 있어야 한다"

    out = BASE / "feature_semantics.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["column", "elliptic_feature_no", "scope", "description"])
        for no, desc in features:
            column, scope = column_and_scope(no)
            writer.writerow([column, no, scope, "" if desc == "-" else desc])
    known = sum(desc != "-" for _, desc in features)
    print(f"saved: {out} ({known}/166 의미 확인)")


if __name__ == "__main__":
    main()
