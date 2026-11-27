"""튜닝한 블렌드의 테스트 두 번째 채점 (한 번만 실행한다).

blend_extras.py의 randomized search에서 전진 검증 PR-AUC 평균이 +0.014 올라(0.779 → 0.793) 팀 규칙(+0.01 이상)에 따라
최종 설정을 바꾼다. 설정과 임계값은 모두 1~34 안에서 정했고, 테스트는 여기서 한 번만 본다.

- 스코어카드: C 0.05 → 0.01, 구간 8 → 12
- LightGBM: 잎 7, 잎당 최소 20건, reg_lambda 0, 학습률 0.1, 트리 300, colsample 0.8, class_weight balanced
- 임계값: 튜닝한 블렌드의 전진 검증 예측(step 15~34)에서 F1 최대 (노트북과 같은 방식)

실행: python blend_tuned_test.py
결과: results/blend_tuned_test.csv, models/m_setting_b_blend_tuned.joblib
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")

# 노트북 셀을 이 스크립트의 전역에 직접 불러온다. (blend_extras를 import하면 병렬 작업 프로세스마다
# 노트북 전체를 다시 불러와 멈춘다.)
_nb = json.load(open("m_setting_b_model.ipynb", encoding="utf-8"))
for _i in [2, 3, 5, 9, 11, 13, 15]:
    exec(compile("".join(_nb["cells"][_i]["source"]), f"m_setting_b_model.ipynb#cell{_i}", "exec"), globals())
    if _i == 2:
        display = lambda *a, **k: None  # noqa: E731
STABLE = [c for c in RAW + cols_of("cnt_", "val_", "age_", "rel_", "par_", "anc_", "p_") + cols_of("pin_", "p2in_")
          if c not in UNSTABLE]
assert len(STABLE) == 149

summary = json.loads((RES / "blend_extras_summary.json").read_text(encoding="utf-8"))
LG_CFG = summary["tuning"]["best_lgbm"]
SC_CFG = summary["tuning"]["best_scorecard"]
LG_CFG = {k: (int(v) if k in ("num_leaves", "min_child_samples", "n_estimators") else float(v)) for k, v in LG_CFG.items()}
make_lg = lgbm(class_weight="balanced", **LG_CFG)


def make_sc(seed=0):
    return make_pipeline(KBinsDiscretizer(n_bins=int(SC_CFG["bins"]), encode="onehot", strategy="quantile",
                                          quantile_method="averaged_inverted_cdf"),
                         LogisticRegression(C=float(SC_CFG["C"]), max_iter=3000))


# 1) 임계값: 1~34 안의 전진 검증 예측으로만
cv = {T: (a + b) / 2 for T, a, b in zip(FOLD_CUTS, run_cv(make_sc, STABLE, seeds=(0,), fill=True).values(),
                                         run_cv(make_lg, STABLE).values())}
cv_score = score(cv)
o = oof(cv)
m = np.isfinite(o)
THR = tune_threshold(Y[m], o[m])
print(f"튜닝 블렌드 전진 검증 PR-AUC 평균 {cv_score['PR-AUC 평균']:.3f}, 임계값 {THR:.3f}")

# 2) 1~34 전체로 학습 → 35~49 한 번 채점
lab = df_all.loc[is_lab, ["txId", "time_step", "y"] + RAW].merge(feats, on="txId")
lab["y"] = lab["y"].astype(int)
train, test = lab[lab["time_step"] <= 34], lab[lab["time_step"] >= 35].reset_index(drop=True)
sc = make_sc().fit(train[STABLE].fillna(-1.0).to_numpy(np.float32), train["y"].values)
lgs = [make_lg(s).fit(train[STABLE].to_numpy(np.float32), train["y"].values) for s in FINAL_SEEDS]
p_sc = sc.predict_proba(test[STABLE].fillna(-1.0).to_numpy(np.float32))[:, 1]
p_lg = np.mean([g.predict_proba(test[STABLE].to_numpy(np.float32))[:, 1] for g in lgs], axis=0)
prob = (p_sc + p_lg) / 2

y, step = test["y"].values, test["time_step"].values
first = pd.read_csv(RES / "m_test_predictions.csv", encoding="utf-8-sig").set_index("txId").loc[test["txId"], "새 모델: 블렌드"].values
first_thr = joblib.load(BASE / "models" / "m_setting_b_blend.joblib")["threshold"]
rows = []
for name, p, th in [("블렌드 (첫 채점, 노트북 설정)", first, first_thr), ("튜닝 블렌드 (두 번째 채점)", prob, THR)]:
    for period, mask in [("35~49", np.ones(len(y), bool)), ("35~42", step < DARK_MARKET_STEP), ("43~49", step >= DARK_MARKET_STEP)]:
        e = evaluate(y[mask], p[mask], th)
        rows.append({"모델": name, "기간": period, "임계값": round(th, 3), "PR-AUC": e["pr_auc"], "F1": e["f1"],
                     "Precision": e["precision"], "Recall": e["recall"], "ROC-AUC": roc_auc_score(y[mask], p[mask]),
                     "잡은 불법": int(((p[mask] >= th) & (y[mask] == 1)).sum())})
out = pd.DataFrame(rows).round(4)
out.to_csv(RES / "blend_tuned_test.csv", index=False, encoding="utf-8-sig")
print(out.to_string(index=False))

joblib.dump({"scorecard": sc, "lgbm": lgs, "columns": STABLE, "threshold": THR, "scorecard_cfg": SC_CFG,
             "lgbm_cfg": LG_CFG, "cv": cv_score}, BASE / "models" / "m_setting_b_blend_tuned.joblib")
