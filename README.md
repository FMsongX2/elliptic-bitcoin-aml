# Elliptic 비트코인 불법 거래 탐지

Elliptic 데이터셋에서 불법 거래를 탐지하는 기계학습 팀 프로젝트의 실험 저장소다. 높은 F1을 재현하는 데서 멈추지 않고, 그 수치가 어떤 피처와 평가 설정에 의존하는지를 분해한다.

진행 기록은 [PROGRESS.md](PROGRESS.md)에 날짜순으로 쌓는다.

## 연구 질문

라벨이 부족한 비트코인 거래 그래프에서, 그래프 구조 정보와 라벨의 양은 불법 거래 탐지 성능에 각각 얼마나 기여하는가?

1. 거래 자체 정보만으로 어디까지 탐지되는가?
2. 그래프 구조 정보(이웃 집계, 직접 설계한 피처, GNN)는 성능을 얼마나 올리는가?
3. 라벨 없이, 또는 라벨 일부만으로 어디까지 가능한가?
4. 시간이 지나 범죄 패턴이 바뀌면 모델은 어떻게 무너지는가?
5. 평가 설정(피처에 담긴 미래 정보, 학습 때 보이는 그래프 범위)은 결론을 얼마나 바꾸는가?

## 지금까지의 결과

공통 설정은 time step 1~34 학습, 35~49 평가, 불법 클래스 기준 지표다.

| 실험 | 내용 | 핵심 결과 |
|---|---|---|
| E1 | 규칙 기반, 얕은 결정 트리 | 도메인 규칙 F1 0.09~0.14, 깊이 2 트리 0.739 |
| E2 | LogReg / Random Forest / LightGBM | LightGBM(165개 피처) F1 0.825, PR-AUC 0.808 |
| E3 | 엣지리스트로 만든 그래프 피처 | 최고 F1 0.827. 기존 집계 피처 위에서는 거의 늘지 않는다 |
| X1 | 피처 범위별 기여도 | 거래 자체 21개 0.354, 주소 통계 72개 0.774, 거래 시점에 계산 가능한 57개 0.557 |
| X2 | 학습·테스트 구간 주소 중복 | 테스트 불법 거래 1,083건 중 학습 구간 주소 공유 5건. 합법 거래는 35~42 구간의 37%가 공유 |
| X3 | GraphSAGE 대 같은 피처의 MLP 대 엣지를 섞은 그래프 | 165개 피처에서 PR-AUC가 MLP 0.675, GraphSAGE 0.736, 섞은 그래프 0.553, LightGBM 0.809 |
| X4 | 주소 단위(Elliptic++)에서 누수 있는 피처와 없는 피처 | 시간 분할을 지키면 LightGBM F1 0.431 대 0.419 |
| X5 | 폐쇄 전후 불법 거래의 피처 변화 | 깊이 2 트리 규칙을 만족하는 불법 거래가 35~42 구간 69.9%, 43~49 구간 0.0% |
| X6 | GAT, GNN 임베딩 + LightGBM | 원 피처 + GAT 임베딩의 PR-AUC가 거래 자체 21개에서 0.419 → 0.584, 설정 B에서 0.594 → 0.612, 설정 A에서 0.808 → 0.714 |

43번째 time step(다크마켓 폐쇄로 해석되는 시점) 이후에는 모든 방법의 F1이 0.1 아래로 떨어진다. X1에서 사후 최적 임계값으로도 F1 0.2를 넘지 못해, 임계값이 아니라 순위의 실패다. X5에서 원인을 확인했다. 폐쇄 전 불법 거래의 대부분은 수명이 짧은 입력 주소에서 나오는데, 폐쇄 후 불법 거래에는 이 유형이 없다.

자세한 분석은 [docs/01_문제재정의_문헌_데이터.md](docs/01_문제재정의_문헌_데이터.md)에 있다.

## 구성

```
common.py                   공통 모듈: 데이터 로드, 시간 분할, 평가, 결과 기록
graph_features.py           엣지리스트 기반 그래프 피처
download_elliptic.py        Elliptic 다운로드와 검증
build_feature_semantics.py  피처 의미 표 생성 (Šafář et al., 2026)
eda_plots.py                E0 EDA
e1_rules.py                 E1 규칙 기반
e2_baseline.py              E2 지도학습 베이스라인
e3_graph_features.py        E3 그래프 피처
x1_feature_scope.py         X1 피처 범위별 기여도
x2_address_overlap.py       X2 주소 중복 (Elliptic++ 필요)
x3_gnn.py                   X3 그래프의 순수 기여 (requirements-gnn.txt)
x4_address_leakage.py       X4 주소 단위 누수 비교 (Elliptic++, 재계산 피처 필요)
x5_drift_profile.py         X5 폐쇄 전후 피처 변화
x6_gat_embedding.py         X6 GAT, GNN 임베딩 + LightGBM (requirements-gnn.txt)
results/                    실험 결과 CSV
figures/                    그림
docs/                       분석 문서
papers/                     참고 문헌 목록
```

`e*`는 팀 계획의 실험 번호를 따르고, `x*`는 평가 설정을 점검하는 추가 실험이다.

## 실행

```bash
pip install -r requirements.txt
python download_elliptic.py        # data/elliptic/
python build_feature_semantics.py  # feature_semantics.csv
python e2_baseline.py
python e3_graph_features.py
python e1_rules.py                 # E2, E3 결과를 참고치로 읽는다
python x1_feature_scope.py
python x2_address_overlap.py       # data/ellipticpp/ 필요
python x3_gnn.py                   # GPU 권장, pip install -r requirements-gnn.txt
python x4_address_leakage.py       # data/ellipticpp/, data/ellipticpp_lr/ 필요
python x5_drift_profile.py
python x6_gat_embedding.py         # GPU 권장, 약 25분
```

## 데이터

데이터는 저장소에 넣지 않는다.

| 데이터 | 받는 곳 | 위치 |
|---|---|---|
| Elliptic | `download_elliptic.py` (Kaggle 또는 PyG 미러) | `data/elliptic/` |
| Elliptic++ | https://github.com/git-disl/EllipticPlusPlus (Google Drive) | `data/ellipticpp/Elliptic++ Dataset/` |
| 누수 제거 Elliptic++ 주소 피처 | https://github.com/DukHaanYumCha/Leakage-Resistant-Elliptic_pp | `data/ellipticpp_lr/` |
| 거래 해시 매핑 | https://www.kaggle.com/datasets/alexbenzik/deanonymized-995-pct-of-elliptic-transactions | `data/elliptic_deanon/` |

Elliptic 데이터셋은 CC BY-NC-ND 4.0이다. 가공한 데이터를 재배포하지 않는다.

## 참고

피처 의미는 Šafář et al. (2026)의 복원 결과를 따른다. 문헌 목록은 [papers/README.md](papers/README.md)에 있다.
