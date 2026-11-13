"""Elliptic Bitcoin 데이터셋 다운로드 스크립트.

1) Kaggle API 인증(~/.kaggle/kaggle.json 또는 KAGGLE_USERNAME/KAGGLE_KEY)이 있으면 Kaggle에서 받는다.
2) 없으면 PyTorch Geometric 공개 미러(data.pyg.org)에서 같은 원본 CSV 3개를 받는다.

실행: python download_elliptic.py
결과: data/elliptic/ 아래에 CSV 3개
"""
import os
import sys
import zipfile
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data" / "elliptic"
FILES = ["elliptic_txs_features.csv", "elliptic_txs_edgelist.csv", "elliptic_txs_classes.csv"]
KAGGLE_DATASET = "ellipticco/elliptic-data-set"
PYG_MIRROR = "https://data.pyg.org/datasets/elliptic"


def has_kaggle_credentials():
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


def download_from_kaggle():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path=DATA_DIR, unzip=True)
    # Kaggle 압축은 elliptic_bitcoin_dataset/ 하위 폴더에 풀리므로 위로 옮긴다
    for f in FILES:
        for found in DATA_DIR.rglob(f):
            if found.parent != DATA_DIR:
                found.replace(DATA_DIR / f)


def _progress(block_num, block_size, total_size):
    if total_size > 0:
        pct = min(block_num * block_size / total_size * 100, 100)
        print(f"\r  {pct:5.1f}%", end="", flush=True)


def download_from_pyg_mirror():
    for f in FILES:
        if (DATA_DIR / f).exists():
            print(f"[skip] {f} 이미 존재")
            continue
        zip_path = DATA_DIR / f"{f}.zip"
        print(f"[download] {f}.zip")
        urllib.request.urlretrieve(f"{PYG_MIRROR}/{f}.zip", zip_path, _progress)
        print()
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(DATA_DIR)
        zip_path.unlink()


def verify():
    features = pd.read_csv(DATA_DIR / FILES[0], header=None)
    edges = pd.read_csv(DATA_DIR / FILES[1])
    classes = pd.read_csv(DATA_DIR / FILES[2])

    print("\n=== 검증 ===")
    print(f"features : {features.shape}  (예상 203769 x 167)")
    print(f"edgelist : {edges.shape}  (예상 234355 x 2)")
    print(f"classes  : {classes.shape}  (예상 203769 x 2)")
    print(f"time step: {features[1].min()} ~ {features[1].max()}")
    print("라벨 분포:")
    print(classes["class"].value_counts().to_string())

    assert features.shape == (203769, 167)
    assert classes.shape[0] == features.shape[0]


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if all((DATA_DIR / f).exists() for f in FILES):
        print("이미 모든 파일이 있습니다. 검증만 진행합니다.")
    elif has_kaggle_credentials():
        print("Kaggle API로 다운로드합니다.")
        download_from_kaggle()
    else:
        print("Kaggle 인증 정보가 없어 PyG 공개 미러에서 다운로드합니다.")
        download_from_pyg_mirror()
    verify()
    print(f"\n완료: {DATA_DIR}")


if __name__ == "__main__":
    sys.exit(main())
