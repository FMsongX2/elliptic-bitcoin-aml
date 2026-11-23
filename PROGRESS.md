# 진행 기록

최신 항목이 위에 온다. 수치의 출처는 `results/`의 CSV다.

## 2026-11-24

### 설정 B의 unsupervised(E6), semi-supervised(E7), 43번째 time step 이후(E8, E9)

전체 보고서는 [docs/02_설정B_unsupervised_semi_supervised.md](docs/02_설정B_unsupervised_semi_supervised.md)다. 보고서의 표는 `report_tables.py`가 `results/`에서 채우고, `python report_tables.py --check`로 맞는지 확인한다.

모든 실험은 설정 B(transaction-time feature 57개, `feature_cols(df, "time")`)로 했다. 분할은 `common.py`를 따른다. train 1-30으로 학습하고, validation 31-34에서 hyperparameter config와 threshold를 고른 뒤, 1-34로 다시 학습해 test 35-49에서 평가했다. PR-AUC는 `average_precision_score`다.

| 방법 | train label | PR-AUC 35-49 | 95% CI | 35-42 | 43-49 |
|---|---|---|---|---|---|
| Random guess | - | 0.065 | - | 0.092 | 0.025 |
| Unsupervised BIRCH | 0% | 0.299 | 0.156-0.407 | 0.399 | 0.023 |
| LightGBM | 1% | 0.489 | 0.256-0.625 | 0.643 | 0.025 |
| Self-training | 1% | 0.544 | 0.300-0.678 | 0.701 | 0.028 |
| LightGBM | 5% | 0.594 | 0.334-0.727 | 0.736 | 0.050 |
| Self-training | 5% | 0.593 | 0.336-0.728 | 0.748 | 0.047 |
| LightGBM | 10% | 0.624 | 0.365-0.755 | 0.750 | 0.070 |
| Self-training | 10% | 0.635 | 0.387-0.760 | 0.765 | 0.072 |
| LightGBM | 100% | 0.593 | 0.332-0.735 | 0.733 | 0.091 |

BIRCH는 seed 3개, 나머지는 seed 6개의 mean이다(`eval_bootstrap.csv`, `eval_seed_ap.csv`). 95% CI는 test time step을 통째로 1,000번 다시 뽑은 bootstrap이다. "label 1%"는 1-34의 label 29,894건 중 1%(299건)만 학습에 공개한다는 뜻이다.

- label이 없으면 약하다. unsupervised detector 8종 중 가장 좋은 BIRCH가 0.299다.
- label은 1%만 있어도 0.49-0.54가 된다. label 5-10%(1,495-2,989건)와 label 100% LightGBM 사이의 우열은 판정할 수 없었다.
- Self-training이 LightGBM보다 낫다고 판정된 것은 label 1%뿐이다(+0.055 ± 0.015). 5%와 10%에서는 판정할 수 없었다.
- 판정 기준: 차이가 t-distribution 95% 기준(seed 6개면 standard error의 2.57배, 3개면 4.30배)을 넘고, bootstrap 95% CI도 0을 포함하지 않아야 "높다"고 쓴다.

#### Data 진단과 preprocessing (`prep_diagnostics.py --setting B`, `prep_research_b.py`)

- train 1-30에서 kurtosis median이 283이다. normal distribution에 가까운 feature는 57개 중 2개다. 37개는 값의 절반 넘게 한 값에 몰려 있다.
- 선행 거래의 block height(`agg_6`, `agg_7`, `agg_9`)는 train과 test의 KS가 0.996-0.998이다. 사실상 시간을 담은 time-proxy feature다.
- distance-based 방법(kNN, Label Spreading)은 Standardization에서 validation PR-AUC가 가장 낮았다. 이후 distance space는 Quantile transform 57개 + PCA-20을 쓴다. validation에서는 다른 transform이 높았지만 test에서 순위가 뒤집혀, test를 보고 정했다(E7b).

#### Unsupervised (E6b-B, `e6b_improved.py --setting B`)

label 없이 1-30의 모든 거래(unknown 포함)로 detector를 학습했다. validation label로 hyperparameter config와 score 방향을 골랐다. 8종, hyperparameter config 240개, seed 3개다.

| detector | preprocessing | validation | test | 35-42 | 43-49 |
|---|---|---|---|---|---|
| BIRCH | std_notp | 0.566 | 0.299 | 0.399 | 0.023 |
| PCA reconstruction | std_notp | 0.640 | 0.177 | 0.272 | 0.023 |
| GMM component | quantile | 0.408 | 0.156 | 0.237 | 0.022 |
| One-Class SVM | std_notp | 0.452 | 0.149 | 0.234 | 0.025 |
| HDBSCAN | quantile | 0.465 | 0.111 | 0.174 | 0.024 |
| Kernel PCA reconstruction | std_notp | 0.486 | 0.098 | 0.148 | 0.024 |
| Isolation Forest | std | 0.408 | 0.097 | 0.149 | 0.021 |
| GMM density | std | 0.507 | 0.080 | 0.116 | 0.020 |

- 8종 모두 validation에서 고른 score 방향이 반대였다. anomaly score가 낮은 거래일수록 illicit이었다.
- validation에서 가장 좋았던 PCA reconstruction(0.640)은 test에서 0.177이다.

#### Semi-supervised (E7, `e7_semi_supervised.py`)

train label의 1, 5, 10, 50, 100%만 공개하고 나머지 label과 unknown 거래를 unlabeled data로 썼다. classifier는 E2의 LightGBM이다. 아래는 seed 3개의 test PR-AUC이고, ± 는 standard deviation이다.

| 방법 | 1% | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| LightGBM | 0.543 ± 0.022 | 0.603 ± 0.033 | 0.627 ± 0.015 | 0.621 ± 0.031 | 0.603 ± 0.010 |
| LightGBM (54개 feature, time-proxy 제외) | 0.537 ± 0.059 | 0.585 ± 0.035 | 0.594 ± 0.025 | 0.602 ± 0.013 | 0.590 ± 0.003 |
| Self-training | 0.589 ± 0.012 | 0.629 ± 0.039 | 0.645 ± 0.024 | 0.643 ± 0.038 | 0.564 ± 0.019 |
| Label Spreading | 0.370 ± 0.126 | 0.571 ± 0.023 | 0.562 ± 0.041 | 0.607 ± 0.036 | 0.629 ± 0.007 |
| Cluster representative (Lec9) | 0.475 ± 0.056 | 0.630 ± 0.031 | 0.643 ± 0.016 | - | - |
| Cluster label propagation (Lec9) | 0.539 ± 0.056 | 0.623 ± 0.011 | 0.618 ± 0.041 | - | - |
| Active learning (uncertainty) | 0.563 ± 0.058 | 0.491 ± 0.068 | 0.549 ± 0.017 | - | - |

- 같은 seed끼리 LightGBM과 비교해 기준을 넘은 것은 셋이다. Label Spreading 100%는 높고, Active learning 10%와 54개 feature 10%는 낮다.
- validation label도 train과 같은 비율만 있다고 보고 threshold를 정하면(E7b), Self-training의 F1은 0.03 안에서 바뀌지만 LightGBM 1%는 0.07 낮아진다.

#### 개선 시도 (E7c, E7d)

- **E7c (`e7c_search.py`):** LightGBM hyperparameter config 32종(base 4종 × 최근 거래 weight 4종 × unsupervised score feature 유무)과 Self-training variant 11종을 rolling-origin validation(1-20→21-24, 1-26→27-30, 1-30→31-34)으로 비교했다. 일관된 개선은 없었다. validation score와 test 순위의 Spearman correlation은 0.42-0.45다.
- **E7d (`e7d_test_time_st.py`):** time step t를 예측할 때 35..t의 unlabeled 거래를 Self-training에 넣었다(t 뒤 거래는 쓰지 않음). 일관된 개선이 없다(10%: 0.638 → 0.642, 5%: 0.633 → 0.600).

#### 신뢰도 평가 (`eval_reliability.py`)

- 대표 방법은 seed 6개로 다시 쟀다. seed 0-2는 `e7c_search.py`, 3-5는 `e7c_confirm.py`다.
- Self-training, LightGBM, Label Spreading의 ensemble은 미래 time step을 쓰지 않는 두 규칙(35..t 안의 rank average, probability mean)으로 만들었다. 둘 다 Self-training보다 낫다고 판정되지 않았다(10%: 0.605, 0.633 대 0.635).
- time step마다 labeled 거래 top-100을 조사하면 Self-training 10%의 recall이 35-42에서 0.558, 43-49에서 0.393이다.
- 예측 probability의 mean은 실제 illicit ratio를 따라가지 못한다(Self-training 10%, 35-42에서 예측 0.052 대 실제 0.092). fixed probability threshold를 오래 쓰기 어렵다.

#### 바로잡은 것

- 처음에는 ensemble을 test 기간 전체(35-49)에서 rank로 바꿔 만들었다. 이렇게 하면 35의 score가 49의 score distribution을 본다. 고친 뒤 "ensemble이 가장 좋다"는 결론은 사라졌다. `e7c_report.py`와 `e7c_confirm.py`에서 이 계산을 지웠다. `tests/test_eval_reliability.py`는 뒤 time step을 바꾸거나 지워도 앞 time step의 score가 같은지 확인한다.
- 직접 짠 Self-training 반복이 마지막 pseudo-label을 더한 뒤 다시 학습하지 않았다. 고친 뒤 scikit-learn `SelfTrainingClassifier`와 예측이 같은지 `tests/test_self_training.py`로 확인한다.

#### 43번째 time step 이후 (`shift_analysis.py`, `minority_analysis.py`, E8, E8b, E9)

- **무엇이 바뀌었나:** illicit 중 single-input 거래(input 수 `local_3`이 train set의 mode)가 1-34의 92.3%에서 43-49의 11.8%로 줄었다. single-input group의 illicit probability는 21.9%에서 0.6%가 되고, multi-input group은 1.7%에서 4.7%로 올랐다. illicit ratio만 바뀌었다면 두 group이 함께 내려가야 하므로, prior shift만으로는 설명되지 않는다.
- **새 label 없이 (E9, `e9_minority_fix.py`, `e9_hedge.py`):** time-proxy를 뺀 54개 feature로 adjustment 6종과 Hedge 2종을 시험했다. 35-49 전체에서 Baseline model(0.599)보다 높은 방법은 없었다. 대부분 폐쇄 뒤를 올리는 대신 폐쇄 전을 내준다(Calibration + EM: 43-49 0.086 → 0.331, 35-42 0.721 → 0.130).
- **새 label을 넣으면 (E8, `e8_shift_recovery.py`, seed 6개):** 1-42의 label 전부로 학습하면 46-49 PR-AUC가 0.212다. 43-45 label을 더하면 418건(그중 illicit 3-6건)에서 0.271, 1,046건에서 0.321, 4,182건 전부에서 0.460이다. 모두 기준보다 높다고 판정됐다.
- **매 time step 다시 학습하면:** 43-49 time step별 PR-AUC mean이 0.175(한 번만 학습)에서 0.439(expanding window)가 된다. 바로 뒤 43-45는 다시 학습해도 0.150이고, 47부터 회복한다. label이 바로 들어온다고 가정한 upper bound다.
- **옛 label도 적을 때 (E8b, `e8b_shift_ssl.py`):** 1-42 label 10%에 43-45 label을 더했다. Self-training과 LightGBM의 차이는 어느 조건에서도 판정할 수 없었다.

#### 이론으로 본 해석 (`theory_checks.py`)

결과를 본 뒤 같은 data에서 확인한 것이라 독립된 증명은 아니다.

- **learning curve (Lec4):** 같은 시기(1-34에서 random으로 떼어 둔 20%)에서는 label 약 3천 건이면 PR-AUC 0.95에 이르지만, 35-42에서는 0.73 안팎에서 멈춘다. 같은 시기 값은 비슷한 거래가 양쪽에 들어가 optimistic이다.
- **Self-training (Lec9):** label 1% model이 probability 0.95 이상으로 illicit이라고 확신한 예측은 같은 시기에서 95.3%, 35-42에서 82.8%가 맞고, 43-49에서는 0.3%만 맞는다.
- **ensemble (Lec7):** Self-training과 LightGBM의 error correlation이 0.89-0.93으로 높다.
- **unsupervised의 score 방향 (Lec9):** train set에서 illicit은 licit보다 10번째 neighbor가 훨씬 가깝다(median 0.06 대 2.22). anomaly score가 낮을수록 illicit인 것과 맞는다.
- **폐쇄 뒤 PR-AUC (Lec3):** illicit ratio가 9.2%에서 2.5%로 줄었지만 ROC-AUC도 떨어졌다(Self-training 10%: 0.931 → 0.802). prior만의 문제는 아니다.

#### 주의

- test 35-49를 여러 번 보면서 실험했다. 수치는 optimistic일 수 있다.
- 성능은 labeled 거래에서 잰 값이다. unlabeled 거래까지 포함한 실제 alert 성능은 아니다.
- E8은 46-49 한 기간에서 잰 값이다. 판정은 seed 변동만 반영한다.

## 2026-11-13

### 환경

- 4060 Ti 서버 `~/elliptic-ml/`에 venv를 만들고 Elliptic 원본을 받아 검증했다(거래 203,769 × 167열, 엣지 234,355).
- torch 2.11.0+cu128, torch_geometric 2.8.1을 설치했다.
- LightGBM을 `n_jobs=-1`(논리 코어 16개)로 돌리면 다른 작업과 겹쳐 한 번 학습에 17분이 걸렸다. `taskset -c 0-7`로 코어를 8개로 제한하면 전체 E2가 수 분 안에 끝난다.

### 재현

팀 코드를 그대로 돌려 E2·E3 수치를 재현했다.

| 모델 | 팀 수치 | 재현 |
|---|---|---|
| E2 LightGBM (all) | 0.825 | 0.824 |
| E2 Random Forest (all) | 0.817 | 0.818 |
| E3 LightGBM (all + 그래프 피처) | 0.827 | 0.827 |

### 문헌 조사

논문 13편의 원문을 읽고 `docs/01_문제재정의_문헌_데이터.md`에 정리했다. 프로젝트 전제를 바꾼 것은 세 편이다.

- Šafář et al. (2026): 피처 의미를 복원했다. 로컬 피처 93개 중 72개는 주소 누적 통계이고, 테스트 구간 이후 시점을 기준으로 계산되어 있다.
- Liang et al. (2026): Elliptic++ 주소 단위에서 누수를 제거하면 Random Forest F1이 0.846에서 0.291로 떨어진다.
- Maganti (2026): 학습 때 테스트 구간 그래프를 보이지 않으면 GraphSAGE(0.689)가 Random Forest(0.821)보다 낮다.

### X1: 피처 범위별 기여도

| 피처셋 | 수 | F1 | PR-AUC | F1 35~42 | F1 43~49 | ROC-AUC 43~49 | 사후 최적 F1 43~49 |
|---|---|---|---|---|---|---|---|
| tx | 21 | 0.354 | 0.416 | 0.428 | 0.049 | 0.598 | 0.065 |
| addr | 72 | 0.774 | 0.783 | 0.876 | 0.028 | 0.647 | 0.077 |
| nbr_in | 36 | 0.187 | 0.177 | 0.237 | 0.065 | 0.689 | 0.100 |
| nbr_out | 36 | 0.666 | 0.673 | 0.774 | 0.061 | 0.690 | 0.086 |
| tx + nbr_in | 57 | 0.557 | 0.599 | 0.679 | 0.084 | 0.840 | 0.183 |
| tx + nbr_in + nbr_out | 93 | 0.646 | 0.699 | 0.760 | 0.061 | 0.832 | 0.168 |
| local (tx + addr) | 93 | 0.780 | 0.789 | 0.879 | 0.032 | 0.652 | 0.078 |
| all | 165 | 0.824 | 0.809 | 0.914 | 0.031 | 0.738 | 0.100 |

- 로컬 피처의 성능은 주소 통계에서 나온다.
- 거래 시점에 계산 가능한 피처(tx + nbr_in)만 쓰면 F1 0.557이다.
- 43~49 구간은 사후 최적 임계값으로도 F1 0.2를 넘지 못한다.

### X2: 주소 중복

- 테스트 구간 입력 주소 149,061개 중 학습 구간에 등장한 주소는 3,164개다.
- 테스트 불법 거래 1,083건 중 학습 구간 주소를 공유하는 것은 5건이다. "모델이 같은 불법 주소를 다시 알아본다"는 가설은 기각했다.
- 합법 거래는 35~42 구간의 37%, 43~49 구간의 26%가 학습 구간 주소를 공유한다.

### X5: 폐쇄 전후 불법 거래의 피처 변화

E1 깊이 2 트리 규칙은 "입력 주소 수명의 최댓값이 작고(`local_52` ≤ -0.488) 고유 출력 주소 수가 적다(`local_13` ≤ -0.053)"이다. 기간·클래스별로 이 조건을 만족하는 비율은 다음과 같다.

| 기간 | 클래스 | 거래 수 | 입력 주소 수명 조건 | 출력 주소 수 조건 | 규칙 전체 |
|---|---|---|---|---|---|
| 1~34 | 불법 | 3,462 | 70.1% | 65.4% | 54.2% |
| 1~34 | 합법 | 26,432 | 11.8% | 9.4% | 0.0% |
| 35~42 | 불법 | 914 | 82.2% | 74.4% | 69.9% |
| 35~42 | 합법 | 9,069 | 13.3% | 2.2% | 0.0% |
| 43~49 | 불법 | 169 | 1.2% | 4.1% | 0.0% |
| 43~49 | 합법 | 6,518 | 22.1% | 2.1% | 0.0% |

- 폐쇄 전 불법 거래의 70~82%는 수명이 짧은 입력 주소에서 나온다. 폐쇄 후에는 1.2%다. 불법 거래의 입력 주소 수명 중앙값은 -0.49에서 0.55로 올라가 합법 거래(0.19)보다 길어진다.
- 폐쇄 전 구간의 탐지 성능은 "수명이 짧은 주소에서 소수의 주소로 보내는 거래"라는 한 가지 유형을 잡는 데서 나왔다. 폐쇄 후에 남은 불법 거래는 이 유형이 아니다.
- 중앙값이 가장 크게 움직인 피처는 후속 거래 수수료의 순서 상관(`agg_52`, `agg_53`), 출력 금액의 순서 상관(`local_19`, `local_20`), 입력 주소 수명(`local_52`, `local_54`)이다.
- 선행 거래의 블록 높이(`agg_7`, `agg_9`)는 시간이 지나면 값이 커지는 피처다. 시점 정보를 그대로 담고 있어 트리 모델이 시기를 구분하는 데 쓸 수 있다.

### X3: 그래프의 순수 기여 (GraphSAGE 대 MLP 대 섞은 그래프)

같은 피처, 같은 깊이와 폭(은닉 128, 2층), 같은 학습 절차에서 연결 정보만 바꿨다. 시드 10개 평균이고 ± 뒤는 표준편차다. `shuffled`는 time step 안에서 엣지의 도착 노드만 섞어 차수를 유지한 그래프다.

| 피처셋 | 모델 | PR-AUC | F1 (0.5) | F1 (검증 임계값) | PR-AUC 35~42 | 사후 최적 F1 43~49 |
|---|---|---|---|---|---|---|
| tx (21) | MLP | 0.452 ± 0.018 | 0.656 ± 0.060 | 0.408 | 0.593 | 0.061 |
| tx (21) | GraphSAGE | 0.519 ± 0.022 | 0.512 ± 0.020 | 0.520 | 0.660 | 0.074 |
| tx (21) | 섞은 그래프 | 0.346 ± 0.015 | 0.583 ± 0.013 | 0.580 | 0.599 | 0.060 |
| tx + nbr_in (57) | MLP | 0.530 ± 0.015 | 0.394 ± 0.042 | 0.452 | 0.615 | 0.071 |
| tx + nbr_in (57) | GraphSAGE | 0.515 ± 0.008 | 0.497 ± 0.009 | 0.491 | 0.603 | 0.083 |
| tx + nbr_in (57) | 섞은 그래프 | 0.472 ± 0.031 | 0.444 ± 0.023 | 0.402 | 0.560 | 0.066 |
| all (165) | MLP | 0.675 ± 0.009 | 0.653 ± 0.016 | 0.706 | 0.805 | 0.087 |
| all (165) | GraphSAGE | 0.736 ± 0.003 | 0.693 ± 0.012 | 0.698 | 0.861 | 0.089 |
| all (165) | 섞은 그래프 | 0.553 ± 0.039 | 0.556 ± 0.032 | 0.585 | 0.673 | 0.081 |

같은 피처의 LightGBM(X1)은 PR-AUC가 tx 0.416, tx + nbr_in 0.599, all 0.809다.

- 임계값 0.5의 F1은 설정마다 크게 흔들린다(tx MLP는 0.656이지만 검증 임계값에서는 0.408). 비교는 PR-AUC로 한다.
- 실제 그래프는 PR-AUC를 tx에서 +0.067, all에서 +0.061 올린다. tx + nbr_in에서는 차이가 없다(-0.015, 시드 표준편차 범위).
- 섞은 그래프는 세 피처셋 모두에서 실제 그래프와 MLP보다 낮다. 이 설정에서는 실제 연결에 정보가 있다. Maganti(2026)의 "섞은 그래프가 더 낫다"는 결과는 재현되지 않았다.
- all에서 GraphSAGE F1 0.693은 Maganti(2026)의 0.689와 같은 수준이다.
- 그래프의 기여(PR-AUC +0.06)보다 모델 종류의 차이가 크다. all에서 LightGBM 0.809, GraphSAGE 0.736, MLP 0.675다.
- 선행 거래 집계 피처가 이미 들어 있으면(tx + nbr_in) GraphSAGE가 더하는 것이 없다. all에서의 이득은 이웃의 주소 통계를 전달받는 데서 나온 것으로 보이지만 따로 검증하지 않았다.
- 43~49 구간은 그래프를 써도 회복되지 않는다.
- 검증 구간(31~34)의 PR-AUC는 테스트 성능과 따로 움직였다. 조기 종료 기준으로 쓰면 MLP가 4 epoch에서 멈추는 등 결과가 불안정해서 epoch 수를 300으로 고정했다.

### X4: 주소 단위에서 누수 있는 피처와 없는 피처

Elliptic++ 라벨 있는 주소 265,352개를 첫 등장 time step으로 나눴다(1~34 학습 172,903개, 35~49 평가 92,449개, 불법 비율 5.4% / 5.3%). 피처만 바꿨다.

| 피처 | 모델 | F1 | PR-AUC | ROC-AUC | F1 35~42 | F1 43~49 | ROC-AUC 43~49 |
|---|---|---|---|---|---|---|---|
| Elliptic++ 배포 피처 (전체 이력 요약) | LightGBM | 0.431 | 0.375 | 0.894 | 0.598 | 0.236 | 0.846 |
| Elliptic++ 배포 피처 (전체 이력 요약) | LogReg | 0.333 | 0.283 | 0.817 | 0.439 | 0.179 | 0.705 |
| 재계산 피처 (Liang et al., 2026) | LightGBM | 0.419 | 0.364 | 0.885 | 0.578 | 0.204 | 0.833 |
| 재계산 피처 (Liang et al., 2026) | LogReg | 0.337 | 0.284 | 0.817 | 0.442 | 0.182 | 0.705 |

- 시간 분할을 지키면 피처 누수의 영향은 F1 0.012로 작다. Liang et al.이 보고한 0.846 → 0.291의 하락은 대부분 분할 방식(위치 기준 → 시간 기준)에서 온다.
- Liang et al.의 Random Forest 수치(시간 분할에서 0.257, 재계산 피처에서 0.291)와 값도 방향도 다르다. 모델이 달라서(LightGBM 500개 트리 대 Random Forest 50개 트리) 직접 비교는 안 된다.
- 주소 단위는 전체 성능이 낮지만 폐쇄 이후에도 순위가 유지된다(43~49 ROC-AUC 0.846). 거래 단위(0.738)와 다르다.

### 데이터 확보

- Elliptic++ (2.1GB), 누수 제거 Elliptic++ 주소 피처(주소 265,352개), 거래 해시 매핑(202,804개)을 서버에 받았다.
- 거래 단위에서 주소 통계를 시점별로 재계산하려면 전체 원장이 필요하다. 서버 디스크 여유(38GB)로는 불가능하다.

### 다음

- X3 후속: all에서 GraphSAGE가 얻는 이득이 이웃의 주소 통계에서 오는지 확인한다(tx + addr 대 tx + nbr 비교). GAT, 엣지 방향을 구분한 집계, GraphSAGE 임베딩 + LightGBM 결합을 추가한다.
- E1: R1(거래 금액) 규칙을 `local_0`과 Elliptic++ `total_BTC`로 구현하고, peel chain을 입출력 금액으로 검증한다.
- E7: 능동 학습(Lorenz et al., 2020)을 라벨 비율별 곡선의 비교군에 넣는다.
- X1~X5 결과를 팀과 공유하고, 진행상황 문서의 전제(피처 비공개, 로컬 피처의 의미, GNN 목표)를 고칠지 정한다.
