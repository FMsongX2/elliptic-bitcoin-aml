"""피처 가공 결과를 시간 기반 교차검증(전진 검증)으로 다시 잰다 (설정 B).

팀원 노트북 m_setting_b_model.ipynb와 같은 틀을 써서 숫자를 바로 비교할 수 있게 한다.
- 폴드 T ∈ {14, 18, 22, 26, 30}: 1~T 학습 → T+1~34 예측
- 예측 거리 h = t - T 를 1~4 / 5~8 / 9~15 로 나눠 PR-AUC, 그 평균이 선택 기준
- F1: 임계값을 그보다 과거 폴드들의 "다음 4 step" 예측에서 정해 적용
- 공식 검증 PR-AUC: T=30 폴드의 31~34 (팀 공통 검증과 같음)

선택은 1~34 안에서만 한다. 테스트(35~49)는 고른 모델 하나만 마지막에 한 번 채점한다.
임계값은 팀원 방식대로 모든 폴드의 "다음 4 step" 예측을 모아 F1 최대로 정한다.

실행: python forward_cv.py
결과: results/forward_cv_summary.csv, results/forward_cv_test.csv, figures/forward_cv_horizon.png
"""
import time
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from sklearn.tree import DecisionTreeClassifier

from common import BASE, DARK_MARKET_STEP, clear_results, evaluate, load_dataset, load_edges, log_result
from fe_experiment import variants
from feature_engineering import build_engineered_features

warnings.filterwarnings("ignore", category=UserWarning)

EXPERIMENT = "FCV"
FOLD_CUTS, NEXT = [14, 18, 22, 26, 30], 4
BUCKETS = [(1, 4), (5, 8), (9, 15)]
SEEDS = (0, 1, 2)
DEV_END = 34
FIG_DIR, RES_DIR = BASE / "figures", BASE / "results"

# 팀원 노트북 결과 (저장된 출력에서 옮김, 여기서 다시 돌리지 않음)
TEAMMATE = {
    "[팀원] 블렌드 (안정 피처 149)": {"PR-AUC h1~4": 0.811, "PR-AUC h5~8": 0.780, "PR-AUC h9~15": 0.751,
                                 "PR-AUC 평균": 0.780, "F1 평균": 0.678, "공식 검증 PR-AUC": 0.779,
                                 "test_pr_auc": 0.641, "test_f1": 0.488},
    "[팀원] 블렌드 엄격 버전": {"PR-AUC h1~4": 0.829, "PR-AUC h5~8": 0.757, "PR-AUC h9~15": 0.766,
                          "PR-AUC 평균": 0.784, "F1 평균": 0.672, "공식 검증 PR-AUC": 0.757,
                          "test_pr_auc": 0.651, "test_f1": 0.601},
}


def tune_threshold(y, prob):
    """F1 최대 임계값. 후보는 예측값 분위수라 확률 눈금이 달라도 쓸 수 있다 (팀원 노트북과 같음)."""
    grid = np.unique(np.quantile(prob, np.linspace(0.30, 0.999, 500)))
    return float(grid[int(np.argmax([f1_score(y, prob >= t, zero_division=0) for t in grid]))])


class FlagRule:
    """0/1 열 하나를 그대로 점수로 쓰는 규칙."""
    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        s = X[:, 0].astype(float)
        return np.column_stack([1 - s, s])


def lgbm(**kw):
    base = dict(n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.5, n_jobs=4, verbose=-1)
    base.update(kw)
    return lambda seed: lgb.LGBMClassifier(random_state=seed, **base)


MODELS = {
    "LightGBM 기본": (lgbm(), SEEDS),
    "LightGBM 규제": (lgbm(n_estimators=200, num_leaves=8, min_child_samples=100, reg_lambda=5.0), SEEDS),
    "결정 트리 깊이 2": (lambda s: DecisionTreeClassifier(max_depth=2, random_state=0), (0,)),
    "결정 트리 깊이 4": (lambda s: DecisionTreeClassifier(max_depth=4, random_state=0), (0,)),
}


def _fit_predict(make, X, y, tr, ev, seed):
    return make(seed).fit(X[tr], y[tr]).predict_proba(X[ev])[:, 1]


def run_cv(dev, make, cols, seeds):
    X, Y, STEP = dev[cols].to_numpy(np.float32), dev["y"].values, dev["time_step"].values
    jobs = [(T, s) for T in FOLD_CUTS for s in seeds]
    res = Parallel(n_jobs=4)(delayed(_fit_predict)(make, X, Y, STEP <= T, STEP > T, s) for T, s in jobs)
    preds = {}
    for T in FOLD_CUTS:
        p = np.full(len(dev), np.nan)
        p[STEP > T] = np.mean([r for (t, _), r in zip(jobs, res) if t == T], axis=0)
        preds[T] = p
    return preds


def score(dev, preds):
    Y, STEP = dev["y"].values, dev["time_step"].values
    nxt = {T: (STEP > T) & (STEP <= T + NEXT) for T in FOLD_CUTS}
    out = {}
    for lo, hi in BUCKETS:
        aps, f1s = [], []
        for T in FOLD_CUTS:
            m = (STEP >= T + lo) & (STEP <= min(T + hi, DEV_END))
            if m.sum() == 0 or Y[m].sum() == 0:
                continue
            aps.append(average_precision_score(Y[m], preds[T][m]))
            past = [t for t in FOLD_CUTS if t < T]
            if past:
                th = tune_threshold(np.concatenate([Y[nxt[t]] for t in past]),
                                    np.concatenate([preds[t][nxt[t]] for t in past]))
                f1s.append(f1_score(Y[m], preds[T][m] >= th, zero_division=0))
        out[f"PR-AUC h{lo}~{hi}"] = float(np.mean(aps))
        out[f"F1 h{lo}~{hi}"] = float(np.mean(f1s))
    out["PR-AUC 평균"] = float(np.mean([out[f"PR-AUC h{lo}~{hi}"] for lo, hi in BUCKETS]))
    out["F1 평균"] = float(np.mean([out[f"F1 h{lo}~{hi}"] for lo, hi in BUCKETS]))
    out["공식 검증 PR-AUC"] = float(average_precision_score(Y[nxt[30]], preds[30][nxt[30]]))
    return out


def oof_threshold(dev, preds):
    """모든 폴드의 '다음 4 step' 예측을 모아 F1 최대 임계값 (팀원 최종 모델과 같은 방식)."""
    Y, STEP = dev["y"].values, dev["time_step"].values
    ys, ps = [], []
    for T in FOLD_CUTS:
        m = (STEP > T) & (STEP <= T + NEXT)
        ys.append(Y[m]), ps.append(preds[T][m])
    return tune_threshold(np.concatenate(ys), np.concatenate(ps))


def main():
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
    clear_results(EXPERIMENT)   # 다시 실행해도 results.csv에 같은 행이 쌓이지 않게
    df_all = load_dataset(labeled_only=False)
    fe = build_engineered_features(df_all, load_edges())
    df = df_all[df_all["y"].notna()].merge(fe, on="txId")
    df["y"] = df["y"].astype(int)
    dev = df[df["time_step"] <= DEV_END].reset_index(drop=True)
    test = df[df["time_step"] > DEV_END].reset_index(drop=True)

    var = variants([c for c in fe.columns if c != "txId"])
    rows, P = [], {}
    t0 = time.time()
    P[("규칙: 입력 1개·출력 1개", "-")] = run_cv(dev, lambda s: FlagRule(), ["fe_1in1out"], (0,))
    for vname, cols in var.items():
        for mname, (make, seeds) in MODELS.items():
            P[(mname, vname)] = run_cv(dev, make, cols, seeds)
        print(f"  done: {vname} ({len(cols)} features) {time.time() - t0:.0f}s")

    for (mname, vname), preds in P.items():
        rows.append({"model": mname, "variant": vname, **score(dev, preds)})
    summary = pd.DataFrame(rows).sort_values("PR-AUC 평균", ascending=False).round(3)
    summary.to_csv(RES_DIR / "forward_cv_summary.csv", index=False, encoding="utf-8-sig")

    show = ["model", "variant", "PR-AUC h1~4", "PR-AUC h5~8", "PR-AUC h9~15", "PR-AUC 평균", "F1 평균",
            "공식 검증 PR-AUC"]
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print("\n=== 전진 검증 (1~34 안에서만, PR-AUC 평균 순) ===")
        print(summary[show].to_string(index=False))
        print("\n참고: 팀원 노트북 (저장된 출력)")
        print(pd.DataFrame(TEAMMATE).T[show[2:]].to_string())

    # 선택: 전진 검증 PR-AUC 평균 1위 (규칙 제외). 테스트는 이 모델 하나만 채점
    best = summary[summary["model"] != "규칙: 입력 1개·출력 1개"].iloc[0]
    key = (best["model"], best["variant"])
    th = oof_threshold(dev, P[key])
    make, seeds = MODELS[key[0]]
    cols = var[key[1]]
    prob = np.mean([make(s).fit(dev[cols], dev["y"]).predict_proba(test[cols])[:, 1] for s in seeds], axis=0)
    y = test["y"].values
    early = test["time_step"].values < DARK_MARKET_STEP
    m = evaluate(y, prob, th)
    p_, r_, _ = precision_recall_curve(y, prob)
    best_possible = float((2 * p_ * r_ / np.clip(p_ + r_, 1e-12, None)).max())
    out = {"model": key[0], "variant": key[1], "cv_pr_auc": best["PR-AUC 평균"], **m,
           "f1_35_42": f1_score(y[early], prob[early] >= th), "f1_43_49": f1_score(y[~early], prob[~early] >= th),
           "f1_if_test_threshold": best_possible}
    pd.DataFrame([out]).round(4).to_csv(RES_DIR / "forward_cv_test.csv", index=False, encoding="utf-8-sig")
    log_result(EXPERIMENT, key[0], key[1], -1, {k: m[k] for k in ["precision", "recall", "f1", "pr_auc"]},
               f"setting=B, forward-CV selected, threshold=oof {th:.3f}")
    print(f"\n=== 선택된 모델 테스트 1회 채점: {key[0]} / {key[1]} ===")
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in out.items()})

    plot(summary)


def plot(summary):
    top = summary.head(8)
    rule = summary[summary["model"].str.startswith("규칙")]
    data = pd.concat([top, rule])
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = np.arange(len(BUCKETS))
    for _, r in data.iterrows():
        vals = [r[f"PR-AUC h{lo}~{hi}"] for lo, hi in BUCKETS]
        ls = "--" if r["model"].startswith("규칙") else "-"
        ax.plot(xs, vals, ls, marker="o", label=f"{r['model']} | {r['variant']}")
    for name, t in TEAMMATE.items():
        ax.plot(xs, [t[f"PR-AUC h{lo}~{hi}"] for lo, hi in BUCKETS], ":", marker="s", color="black", alpha=0.6,
                label=name)
    ax.set_xticks(xs, [f"{lo}~{hi} step 뒤" for lo, hi in BUCKETS])
    ax.set_ylabel("불법 PR-AUC")
    ax.set_ylim(0.3, 0.9)
    ax.set_title("[설정 B] 전진 검증: 예측 거리별 PR-AUC (1~34 안에서)", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e5e0")
    ax.legend(fontsize=7, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "forward_cv_horizon.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
