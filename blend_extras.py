"""최종 모델(블렌드) 후속 실험: 튜닝, 불균형 처리, 임계값 방식, 반지도, 해석과 오류 분석.

블렌드 노트북(m_setting_b_model.ipynb)의 코드 셀을 그대로 불러 같은 피처·같은 전진 검증 틀을 쓴다.
먼저 노트북을 한 번 실행해 models/m_setting_b_blend.joblib, results/m_test_predictions.csv 를 만들어 둔다.

규칙 (남은일정_한계_1125.md)
- 모든 비교와 선택은 1~34 안의 전진 검증 점수로만 한다. 테스트(35~49)는 보지 않는다.
- 전진 검증에서 지금 설정보다 확실히 나은 것(PR-AUC 평균 +0.01 이상, 임계값은 F1 평균 +0.01 이상)이
  있을 때만 바꾸고, 그때만 테스트를 한 번 더 채점한다("두 번째 채점").
- 해석(SHAP)과 오류 분석은 이미 채점한 최종 모델의 테스트 예측을 설명하는 것이라 선택에 쓰지 않는다.

실행: python blend_extras.py   (약 10~20분)
결과: results/blend_*.csv, results/blend_extras_summary.json, figures/blend_*.png
"""
import json
import time

import matplotlib

matplotlib.use("Agg")

NB_PATH = "m_setting_b_model.ipynb"
NB_CELLS = [2, 3, 5, 9, 11, 13, 15, 18, 21]   # 준비, 데이터, 피처, 안정성 점검, 피처셋, 전진 검증, 기준 모델, 피처셋 비교, 최종 구성 모델
GAIN_MIN = 0.01


def _load_notebook():
    nb = json.load(open(NB_PATH, encoding="utf-8"))
    for i in NB_CELLS:
        exec(compile("".join(nb["cells"][i]["source"]), f"{NB_PATH}#cell{i}", "exec"), globals())
        if i == 2:
            globals()["display"] = lambda *a, **k: None


t_all = time.time()
_load_notebook()
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})

import joblib  # noqa: E402

BASE_BLEND = {T: (P["스코어카드"][T] + P["LightGBM 클래스 가중"][T]) / 2 for T in FOLD_CUTS}
BASE_SCORE = score(BASE_BLEND)
SUMMARY = {"base_blend_cv": BASE_SCORE}
print(f"[재현] 블렌드 전진 검증 PR-AUC 평균 {BASE_SCORE['PR-AUC 평균']:.3f} (노트북 0.780)")


def blend_of(p_sc, p_lg):
    return {T: (p_sc[T] + p_lg[T]) / 2 for T in FOLD_CUTS}


def row(name, s, **kw):
    return {"이름": name, **kw, **{k: round(v, 4) for k, v in s.items()}}


# ---------------------------------------------------------------- 1. 하이퍼파라미터 튜닝 (randomized search)
def tuning():
    rng = np.random.default_rng(0)
    space = {"num_leaves": [7, 15, 31, 63], "min_child_samples": [20, 40, 100, 200],
             "reg_lambda": [0.0, 1.0, 5.0, 10.0], "learning_rate": [0.02, 0.03, 0.05, 0.1],
             "n_estimators": [150, 300, 500], "colsample_bytree": [0.3, 0.5, 0.8]}
    current = dict(num_leaves=15, min_child_samples=40, reg_lambda=5.0, learning_rate=0.03, n_estimators=300,
                   colsample_bytree=0.5)
    configs = [current] + [{k: v[rng.integers(len(v))] for k, v in space.items()} for _ in range(30)]
    rows, preds = [], {}
    for i, cfg in enumerate(configs):
        p = run_cv(lgbm(class_weight="balanced", **cfg), STABLE)
        preds[i] = p
        rows.append(row("현재 설정" if i == 0 else f"후보 {i}", score(blend_of(P["스코어카드"], p)),
                        모델="LightGBM", **{k: v for k, v in cfg.items()},
                        단독_PR_AUC=round(score(p)["PR-AUC 평균"], 4)))
    print(f"  LightGBM {len(configs)}개 완료")

    sc_rows, sc_preds = [], {}
    for C in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        for bins in [5, 8, 12]:
            def make(seed, C=C, bins=bins):
                return make_pipeline(KBinsDiscretizer(n_bins=bins, encode="onehot", strategy="quantile",
                                                      quantile_method="averaged_inverted_cdf"),
                                     LogisticRegression(C=C, max_iter=3000))
            p = run_cv(make, STABLE, seeds=(0,), fill=True)
            sc_preds[(C, bins)] = p
            sc_rows.append(row("현재 설정" if (C, bins) == (0.05, 8) else f"C={C}, 구간 {bins}",
                               score(blend_of(p, P["LightGBM 클래스 가중"])), 모델="스코어카드", C=C, 구간=bins,
                               단독_PR_AUC=round(score(p)["PR-AUC 평균"], 4)))
    print("  스코어카드 18개 완료")

    lg = pd.DataFrame(rows).sort_values("PR-AUC 평균", ascending=False)
    sc = pd.DataFrame(sc_rows).sort_values("PR-AUC 평균", ascending=False)
    best_lg = configs[int(lg.iloc[0]["이름"].split()[-1]) if lg.iloc[0]["이름"] != "현재 설정" else 0]
    best_sc = (sc.iloc[0]["C"], int(sc.iloc[0]["구간"]))
    i_lg = configs.index(best_lg)
    both = score(blend_of(sc_preds[best_sc], preds[i_lg]))
    out = pd.concat([lg, sc], ignore_index=True)
    out.to_csv(RES / "blend_tuning.csv", index=False, encoding="utf-8-sig")
    gain = both["PR-AUC 평균"] - BASE_SCORE["PR-AUC 평균"]
    res = {"best_lgbm": best_lg, "best_scorecard": {"C": best_sc[0], "bins": best_sc[1]},
           "best_both_cv": both, "gain": gain, "adopt": bool(gain >= GAIN_MIN),
           "lgbm_range": [float(lg["PR-AUC 평균"].min()), float(lg["PR-AUC 평균"].max())],
           "sc_range": [float(sc["PR-AUC 평균"].min()), float(sc["PR-AUC 평균"].max())]}
    print(f"[튜닝] 최고 조합 PR-AUC 평균 {both['PR-AUC 평균']:.3f} (현재 {BASE_SCORE['PR-AUC 평균']:.3f}, "
          f"차이 {gain:+.3f}) → {'바꿈' if res['adopt'] else '유지'}")
    print(f"       LightGBM 31개 범위 {res['lgbm_range'][0]:.3f}~{res['lgbm_range'][1]:.3f}, "
          f"스코어카드 18개 범위 {res['sc_range'][0]:.3f}~{res['sc_range'][1]:.3f}")
    return res


# ---------------------------------------------------------------- 2. 불균형 처리 (E5)
class UnderSampled:
    """음성(합법)을 양성 수의 ratio배만 무작위로 남겨 학습한다."""
    def __init__(self, seed, ratio):
        self.seed, self.ratio = seed, ratio
        self.model = lgbm(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                          reg_lambda=5.0)(seed)

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        keep = np.concatenate([pos, rng.choice(neg, min(len(neg), self.ratio * len(pos)), replace=False)])
        self.model.fit(X[keep], y[keep])
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def imbalance():
    base = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40, reg_lambda=5.0)
    ratio = float((Y == 0).sum() / (Y == 1).sum())
    variants = {
        "클래스 가중 balanced (현재)": P["LightGBM 클래스 가중"],
        "가중 없음": run_cv(lgbm(**base), STABLE),
        f"가중 √(음성/양성) = {ratio ** 0.5:.1f}": run_cv(lgbm(scale_pos_weight=ratio ** 0.5, **base), STABLE),
        "언더샘플링 1:1": run_cv(lambda s: UnderSampled(s, 1), STABLE),
        "언더샘플링 1:3": run_cv(lambda s: UnderSampled(s, 3), STABLE),
    }
    rows = []
    for name, p in variants.items():
        rows.append(row(name, score(blend_of(P["스코어카드"], p)), 단독_PR_AUC=round(score(p)["PR-AUC 평균"], 4)))
    out = pd.DataFrame(rows).sort_values("PR-AUC 평균", ascending=False)
    out.to_csv(RES / "blend_imbalance.csv", index=False, encoding="utf-8-sig")
    best = out.iloc[0]
    gain = best["PR-AUC 평균"] - BASE_SCORE["PR-AUC 평균"]
    print(f"[불균형] 최고 '{best['이름']}' {best['PR-AUC 평균']:.3f} (현재 {BASE_SCORE['PR-AUC 평균']:.3f}, 차이 {gain:+.3f})")
    return {"table": out[["이름", "PR-AUC 평균", "F1 평균", "단독_PR_AUC"]].to_dict(orient="records"),
            "adopt": bool(gain >= GAIN_MIN and best["이름"] != "클래스 가중 balanced (현재)")}


# ---------------------------------------------------------------- 3. 임계값 방식
K_GRID = np.round(np.arange(0.005, 0.30, 0.005), 3)


def flag_top_k(prob, steps, k):
    """time step마다 점수 상위 k 비율을 경보로 낸다. 같은 step 안의 순위만 쓴다."""
    out = np.zeros(len(prob), bool)
    for t in np.unique(steps):
        m = steps == t
        n = max(1, int(round(k * m.sum())))
        idx = np.flatnonzero(m)[np.argsort(-prob[m])[:n]]
        out[idx] = True
    return out


def choose_k(ys, ps, ss):
    f1s = [f1_score(ys, flag_top_k(ps, ss, k), zero_division=0) for k in K_GRID]
    return float(K_GRID[int(np.argmax(f1s))])


def threshold_methods(preds):
    """세 방식의 F1을 예측 거리별로 잰다. 기준값은 모두 그보다 과거 폴드에서만 정한다."""
    res = {"확률 임계값 (현재)": [], "time step별 상위 k%": [], "고정 경보율 (과거 불법 비율)": []}
    for lo, hi in BUCKETS:
        acc = {k: [] for k in res}
        for T in FOLD_CUTS:
            m = (STEP >= T + lo) & (STEP <= min(T + hi, 34))
            past = [t for t in FOLD_CUTS if t < T]
            if not past or m.sum() == 0 or Y[m].sum() == 0:
                continue
            pm = np.concatenate([FOLDS[t]["next"].nonzero()[0] for t in past])
            th = tune_threshold(Y[pm], np.concatenate([preds[t][FOLDS[t]["next"]] for t in past]))
            acc["확률 임계값 (현재)"].append(f1_score(Y[m], preds[T][m] >= th, zero_division=0))
            ys = np.concatenate([Y[FOLDS[t]["next"]] for t in past])
            ps = np.concatenate([preds[t][FOLDS[t]["next"]] for t in past])
            ss = np.concatenate([STEP[FOLDS[t]["next"]] for t in past])
            k = choose_k(ys, ps, ss)
            acc["time step별 상위 k%"].append(f1_score(Y[m], flag_top_k(preds[T][m], STEP[m], k), zero_division=0))
            rate = float(Y[STEP <= T].mean())     # 학습 구간의 불법 비율만큼 경보
            acc["고정 경보율 (과거 불법 비율)"].append(
                f1_score(Y[m], flag_top_k(preds[T][m], STEP[m], rate), zero_division=0))
        for k in res:
            res[k].append(float(np.mean(acc[k])))
    rows = [{"방식": k, "F1 h1~4": v[0], "F1 h5~8": v[1], "F1 h9~15": v[2], "F1 평균": float(np.mean(v))}
            for k, v in res.items()]
    out = pd.DataFrame(rows).round(4)
    out.to_csv(RES / "blend_threshold.csv", index=False, encoding="utf-8-sig")
    base = out.loc[out["방식"] == "확률 임계값 (현재)", "F1 평균"].item()
    best = out.sort_values("F1 평균", ascending=False).iloc[0]
    print(f"[임계값] 최고 '{best['방식']}' F1 평균 {best['F1 평균']:.3f} (현재 {base:.3f})")
    return {"table": out.to_dict(orient="records"), "best": best["방식"],
            "adopt": bool(best["방식"] != "확률 임계값 (현재)" and best["F1 평균"] - base >= GAIN_MIN)}


# ---------------------------------------------------------------- 4. 반지도: self-training (E7)
def self_training(hi=0.9, lo=0.02):
    unl = df_all.loc[~is_lab, ["txId", "time_step"] + RAW].merge(feats, on="txId").reset_index(drop=True)
    U_step = unl["time_step"].values
    Xd, Xdf = dev[STABLE].to_numpy(np.float32), dev_f[STABLE].to_numpy(np.float32)
    Xu, Xuf = unl[STABLE].to_numpy(np.float32), unl[STABLE].fillna(-1.0).to_numpy(np.float32)
    preds, added = {}, {}

    def fit_pred(Xa, Xaf, ya, ev):
        sc = make_scorecard().fit(Xaf, ya)
        lgs = [LGBM_CW(s).fit(Xa, ya) for s in SEEDS]

        def predict(X, Xf):
            return (sc.predict_proba(Xf)[:, 1] + np.mean([m.predict_proba(X)[:, 1] for m in lgs], axis=0)) / 2
        return predict

    for T in FOLD_CUTS:
        tr, ev = FOLDS[T]["train"], FOLDS[T]["long"]
        um = U_step <= T                                   # 학습 구간과 같은 시기의 라벨 없는 거래만
        pred0 = fit_pred(Xd[tr], Xdf[tr], Y[tr], ev)
        pu = pred0(Xu[um], Xuf[um])
        pos, neg = pu >= hi, pu <= lo
        Xa = np.vstack([Xd[tr], Xu[um][pos | neg]])
        Xaf = np.vstack([Xdf[tr], Xuf[um][pos | neg]])
        ya = np.concatenate([Y[tr], pos[pos | neg].astype(int)])
        pred1 = fit_pred(Xa, Xaf, ya, ev)
        p = np.full(len(dev), np.nan)
        p[ev] = pred1(Xd[ev], Xdf[ev])
        preds[T] = p
        added[T] = {"가짜 불법": int(pos.sum()), "가짜 합법": int(neg.sum()), "라벨": int(tr.sum())}
    s = score(preds)
    gain = s["PR-AUC 평균"] - BASE_SCORE["PR-AUC 평균"]
    pd.DataFrame([row("블렌드 (현재)", BASE_SCORE), row(f"self-training (≥{hi} 불법, ≤{lo} 합법)", s)]).to_csv(
        RES / "blend_selftrain.csv", index=False, encoding="utf-8-sig")
    print(f"[반지도] self-training PR-AUC 평균 {s['PR-AUC 평균']:.3f} (현재 {BASE_SCORE['PR-AUC 평균']:.3f}, 차이 {gain:+.3f})")
    return {"cv": s, "gain": gain, "added": added, "adopt": bool(gain >= GAIN_MIN)}


# ---------------------------------------------------------------- 5. 해석(SHAP)과 오류 분석 (최종 모델, 선택에 쓰지 않음)
def explain_and_errors():
    saved = joblib.load(BASE / "models" / "m_setting_b_blend.joblib")
    cols, thr = saved["columns"], saved["threshold"]
    lab_all = df_all.loc[is_lab, ["txId", "time_step", "y"] + RAW].merge(feats, on="txId")
    test = lab_all[lab_all["time_step"] >= 35].reset_index(drop=True)
    X = test[cols].to_numpy(np.float32)
    # LightGBM 내장 TreeSHAP (pred_contrib): 마지막 열은 기준값
    contrib = np.mean([m.booster_.predict(X, pred_contrib=True)[:, :-1] for m in saved["lgbm"]], axis=0)
    early = test["time_step"].values < DARK_MARKET_STEP
    sem = pd.read_csv(BASE / "feature_semantics.csv").set_index("column")["description"]
    imp = pd.DataFrame({"피처": cols,
                        "평균 |SHAP| 35~49": np.abs(contrib).mean(0),
                        "평균 |SHAP| 35~42": np.abs(contrib[early]).mean(0),
                        "평균 |SHAP| 43~49": np.abs(contrib[~early]).mean(0),
                        "불법 쪽 평균 SHAP (불법 거래)": contrib[test["y"].values == 1].mean(0)})
    imp["의미"] = [sem.get(c, "") for c in cols]
    imp = imp.sort_values("평균 |SHAP| 35~49", ascending=False).reset_index(drop=True)
    imp.round(4).to_csv(RES / "blend_shap_top.csv", index=False, encoding="utf-8-sig")

    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(top))
    ax.barh(y + 0.2, top["평균 |SHAP| 35~42"], 0.4, color="#2a78d6", label="35~42 (폐쇄 전)")
    ax.barh(y - 0.2, top["평균 |SHAP| 43~49"], 0.4, color="#eb6834", label="43~49 (폐쇄 후)")
    ax.set_yticks(y, top["피처"])
    ax.set_xlabel("평균 |SHAP| (LightGBM 쪽, 로그 오즈)")
    ax.set_title("최종 블렌드의 LightGBM이 크게 본 피처 15개 (테스트 거래)", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "blend_shap.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # 오류 분석: 노트북이 저장한 최종 예측과 고정 임계값
    pred = pd.read_csv(RES / "m_test_predictions.csv", encoding="utf-8-sig")
    d = test[["txId", "time_step", "y", "cnt_n_in", "cnt_n_out"]].merge(pred[["txId", "새 모델: 블렌드"]], on="txId")
    d["경보"] = d["새 모델: 블렌드"] >= thr
    d["기간"] = np.where(d["time_step"] < DARK_MARKET_STEP, "35~42", "43~49")
    n_in, n_out = d["cnt_n_in"], d["cnt_n_out"]
    d["모양"] = np.select([(n_in == 1) & (n_out == 1), (n_in == 1) & (n_out == 2), (n_in >= 3) & (n_out == 1),
                         n_out >= 3], ["입력1·출력1", "입력1·출력2", "여러 입력→출력1", "출력 3개 이상"], "기타")
    d["결과"] = np.select([d["y"].eq(1) & d["경보"], d["y"].eq(1) & ~d["경보"], d["y"].eq(0) & d["경보"]],
                        ["TP (잡음)", "FN (놓침)", "FP (오탐)"], "TN")
    cm = d.pivot_table(index="기간", columns="결과", values="txId", aggfunc="count", fill_value=0)
    by_shape = d[d["결과"] != "TN"].pivot_table(index=["기간", "모양"], columns="결과", values="txId",
                                                aggfunc="count", fill_value=0)
    pd.concat({"혼동 행렬": cm}, names=["표"]).to_csv(RES / "blend_errors_confusion.csv", encoding="utf-8-sig")
    by_shape.to_csv(RES / "blend_errors_by_shape.csv", encoding="utf-8-sig")
    print("[오류] 혼동 행렬\n", cm.to_string())
    print("[오류] 모양별\n", by_shape.to_string())
    print("[해석] SHAP 상위 10\n", imp.head(10)[["피처", "평균 |SHAP| 35~42", "평균 |SHAP| 43~49", "의미"]].to_string())
    return {"confusion": cm.to_dict(), "by_shape": {f"{a} | {b}": r.to_dict() for (a, b), r in by_shape.iterrows()},
            "shap_top10": imp.head(10)[["피처", "평균 |SHAP| 35~49", "평균 |SHAP| 35~42", "평균 |SHAP| 43~49", "의미"]]
            .round(4).to_dict(orient="records")}


if __name__ == "__main__":
    SUMMARY["tuning"] = tuning()
    SUMMARY["imbalance"] = imbalance()
    SUMMARY["threshold"] = threshold_methods(BASE_BLEND)
    SUMMARY["self_training"] = self_training()
    SUMMARY["explain"] = explain_and_errors()
    SUMMARY["any_change"] = any(SUMMARY[k]["adopt"] for k in ["tuning", "imbalance", "threshold", "self_training"])
    SUMMARY["runtime_min"] = round((time.time() - t_all) / 60, 1)
    (RES / "blend_extras_summary.json").write_text(json.dumps(SUMMARY, ensure_ascii=False, indent=2, default=float),
                                                   encoding="utf-8")
    print(f"\n최종 모델을 바꿀 결과가 있나: {SUMMARY['any_change']}  |  {SUMMARY['runtime_min']}분")
