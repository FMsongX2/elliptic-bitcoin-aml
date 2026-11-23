"""Technical terms in the report are written in English (failure record: translated course terms).

The user studies the course in English. Sentences stay Korean; terms must not be translated, in the
prose, in the generated tables, or in the text inside the figures.
"""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TRANSLATED = [
    "비지도", "반지도", "셀프 트레이닝", "라벨", "군집", "능동 학습", "앙상블", "피처", "전처리", "표준화", "퀀타일",
    "첨도", "중앙값", "정규분포", "극단값", "최빈값", "표준오차", "표준편차", "조화 평균", "상관", "편향",
    "정밀도", "재현율", "오탐", "임계값", "시드", "시점", "구간", "학습 곡선", "학습 데이터", "재학습", "검증",
    "테스트", "모델", "분류기", "탐지기", "기준선", "분포", "공변량", "개념 변화", "사전확률", "이상 탐지", "이상 점수",
    "밀도", "재구성", "주성분", "가중치", "반감기", "확률", "오차", "순위", "불확실성", "무작위", "데이터", "표본",
    "이웃", "그래프", "경보", "클래스", "가설", "난수",
]
KEEP = re.compile(r"```.*?```|`[^`\n]*`|\]\([^)]*\)", re.S)


def translated_terms(text):
    plain = KEEP.sub(" ", text)
    return sorted({term for term in TRANSLATED if term in plain})


def test_report_uses_english_terms():
    reports = list((ROOT / "docs").glob("02_설정B_unsupervised_semi_supervised.md"))
    if not reports:
        pytest.skip("report not shipped")
    assert translated_terms(reports[0].read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("name", ["report_tables.py", "report_figures.py"])
def test_generators_use_english_terms(name):
    assert translated_terms((ROOT / name).read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("sentence", ["정밀도가 높다", "셀프 트레이닝은 이득이다", "표준오차의 2배"])
def test_detector_catches_translated_terms(sentence):
    assert translated_terms(sentence)
