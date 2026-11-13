"""E1: 규칙 기반 베이스라인.

사람이 만든 규칙(도메인 지식)과 데이터로 찾은 단순 규칙(얕은 결정 트리)이
머신러닝(LightGBM, E2·E3)과 비교해 얼마나 잡는지 본다.

규칙 (값은 graph_features.py, 엣지리스트로 계산해 의미를 아는 값만 사용)
- R2 분할           : out-degree >= k   (팀원 아이디어 "UTXO 분할"의 도메인 방향)
- R2' 분할 반대     : out-degree <= k   (데이터가 보여 준 방향)
- R3 다중 경유      : peel chain 유사 패턴 길이 >= L
- R4 조합           : R3 AND R2'
- R1 대량 이동      : 금액이 비공개라 보류 (Elliptic++ 확보 시 추가)

임계값은 학습 구간(1~34)에서 불법 F1이 최대가 되도록 고르고, 테스트(35~49)에 고정 적용한다.
얕은 결정 트리도 1~34로 학습하고, 확률 임계값을 같은 방식으로 학습 구간에서 고른다.

실행: python e1_rules.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier, plot_tree

from common import (BASE, DARK_MARKET_STEP, RESULTS_PATH, SEED, clear_results, evaluate,
                    feature_cols, load_dataset, load_edges, log_result, split, tune_threshold)
from graph_features import STRUCT_PREFIX, build_graph_features

EXPERIMENT = "E1"
FIG_DIR = BASE / "figures"
RES_DIR = BASE / "results"

CATEGORY_COLORS = {"기준": "#c3c2b7", "도메인 규칙": "#eb6834", "결정 트리": "#1baf7a",
                   "머신러닝": "#2a78d6"}


def best_threshold(score, y, candidates):
    """score >= t 를 불법으로 볼 때 F1이 최대인 t."""
    f1s = [f1_score(y, score >= t, zero_division=0) for t in candidates]
    i = int(np.argmax(f1s))
    return candidates[i], f1s[i]


def period_f1(test, pred):
    early = test["time_step"].values < DARK_MARKET_STEP
    y = test["y"].values
    return (f1_score(y[early], pred[early], zero_division=0),
            f1_score(y[~early], pred[~early], zero_division=0))


def degree_table(trainval):
    rows = []
    for col, name in [("g_out_deg", "out-degree (나가는 연결)"), ("g_in_deg", "in-degree (들어오는 연결)")]:
        bins = pd.cut(trainval[col], [-1, 0, 1, 2, 3, 5, 10, 1e9],
                      labels=["0", "1", "2", "3", "4-5", "6-10", "11+"])
        t = trainval.groupby(bins, observed=True)["y"].agg(["count", "mean"])
        t.insert(0, "feature", name)
        rows.append(t.reset_index(names="degree"))
    return pd.concat(rows)


def main():
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
    clear_results(EXPERIMENT)

    df_all = load_dataset(labeled_only=False)
    gf = build_graph_features(df_all, load_edges())
    struct_cols = [c for c in gf.columns if c.startswith(STRUCT_PREFIX)]
    df = df_all[df_all["y"].notna()].merge(gf[["txId"] + struct_cols], on="txId")
    df["y"] = df["y"].astype(int)
    train, val, test = split(df)
    trainval = pd.concat([train, val])
    ytv, yte = trainval["y"].values, test["y"].values

    degree_table(trainval).to_csv(RES_DIR / "e1_degree_illicit_rate.csv", index=False,
                                  encoding="utf-8-sig")

    rows = []

    def record(name, category, rule, pred, score, params):
        m = evaluate(yte, score, threshold=0.5) if score is not None else None
        metrics = {
            "precision": evaluate(yte, pred.astype(float))["precision"],
            "recall": evaluate(yte, pred.astype(float))["recall"],
            "f1": f1_score(yte, pred, zero_division=0),
            "pr_auc": m["pr_auc"] if m else evaluate(yte, pred.astype(float))["pr_auc"],
        }
        f1_early, f1_late = period_f1(test, pred)
        log_result(EXPERIMENT, name, "struct" if "트리(all" not in name else "all", SEED,
                   metrics, rule)
        rows.append({"name": name, "category": category, "rule": rule, **metrics,
                     "f1_35_42": f1_early, "f1_43_49": f1_late, "flag_rate": pred.mean(), **params})

    # 기준: 전부 불법이라고 찍기
    record("모두 불법", "기준", "모든 거래 = 불법", np.ones_like(yte, dtype=bool), None, {})

    out_tv, out_te = trainval["g_out_deg"].values, test["g_out_deg"].values
    peel_tv, peel_te = trainval["g_peel_chain"].values, test["g_peel_chain"].values
    deg_cands = np.arange(0, 21)
    peel_cands = np.array([1, 2, 3, 5, 10, 20, 50, 100, 200, 500])

    # R2 분할 (도메인 방향): out-degree >= k
    k, _ = best_threshold(out_tv, ytv, deg_cands[1:])
    record("R2 분할", "도메인 규칙", f"out-degree >= {k}", out_te >= k, out_te, {"k": k})

    # R2' 분할 반대 (데이터 방향): out-degree <= k  ->  -out >= -k
    k2, _ = best_threshold(-out_tv, ytv, -deg_cands)
    k2 = -k2
    record("R2' 분할 반대", "도메인 규칙", f"out-degree <= {k2}", out_te <= k2, -out_te, {"k": k2})

    # R3 다중 경유: peel chain >= L
    L, _ = best_threshold(peel_tv, ytv, peel_cands)
    record("R3 다중 경유", "도메인 규칙", f"peel chain >= {L}", peel_te >= L, peel_te, {"L": L})

    # R4 조합: R3 AND R2'
    best = (-1, None, None)
    for kk in deg_cands:
        for ll in peel_cands:
            f = f1_score(ytv, (peel_tv >= ll) & (out_tv <= kk), zero_division=0)
            if f > best[0]:
                best = (f, kk, ll)
    _, k4, L4 = best
    record("R4 조합", "도메인 규칙", f"peel chain >= {L4} AND out-degree <= {k4}",
           (peel_te >= L4) & (out_te <= k4), None, {"k": k4, "L": L4})

    # 얕은 결정 트리: 데이터가 스스로 찾은 규칙
    trees = {}
    all_cols = feature_cols(df, "all")
    for cols_name, cols in [("구조", struct_cols), ("all", all_cols)]:
        for depth in [1, 2, 3]:
            tree = DecisionTreeClassifier(max_depth=depth, random_state=SEED)
            tree.fit(trainval[cols], ytv)
            th = tune_threshold(ytv, tree.predict_proba(trainval[cols])[:, 1])
            prob = tree.predict_proba(test[cols])[:, 1]
            record(f"트리({cols_name}) 깊이 {depth}", "결정 트리",
                   f"DecisionTree depth={depth}, threshold={th:.2f}", prob >= th, prob,
                   {"depth": depth, "threshold": th})
            trees[(cols_name, depth)] = (tree, cols)

    res = pd.DataFrame(rows)

    # 머신러닝 참고치 (E2, E3 결과에서 읽어 온다)
    ref = pd.read_csv(RESULTS_PATH, encoding="utf-8-sig")
    for exp, fs, label in [("E2", "all", "LightGBM (E2, all)"),
                           ("E3", "all+struct+nbr+nbr2", "LightGBM (E3, 최고)")]:
        r = ref[(ref["experiment"] == exp) & (ref["model"] == "LightGBM")
                & (ref["feature_set"] == fs) & (ref["note"] == "threshold=0.5")]
        if len(r):
            res = pd.concat([res, pd.DataFrame([{
                "name": label, "category": "머신러닝", "rule": "참고치",
                **r[["precision", "recall", "f1", "pr_auc"]].mean().to_dict()}])], ignore_index=True)

    res.to_csv(RES_DIR / "e1_summary.csv", index=False, encoding="utf-8-sig")
    plot_comparison(res)
    plot_trees(trees)

    print("\n=== E1 테스트 결과 (35~49, 임계값은 1~34에서 선택) ===")
    cols = ["name", "rule", "precision", "recall", "f1", "pr_auc", "f1_35_42", "f1_43_49", "flag_rate"]
    with pd.option_context("display.width", 220, "display.max_columns", 20,
                           "display.max_colwidth", 45):
        print(res[cols].round(3).to_string(index=False))
    print("\n학습 구간(1~34) 차수별 불법 비율")
    print(degree_table(trainval).round(3).to_string(index=False))


def plot_comparison(res):
    order = res.iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6.5))
    colors = [CATEGORY_COLORS[c] for c in order["category"]]
    ax.barh(order["name"], order["f1"], color=colors, height=0.65)
    for i, v in enumerate(order["f1"]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("테스트 불법 F1 (35~49)")
    ax.set_title("규칙 기반 vs 결정 트리 vs 머신러닝", loc="left", fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CATEGORY_COLORS.values()]
    ax.legend(handles, CATEGORY_COLORS.keys(), loc="upper right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e6e5e0")
    ax.set_axisbelow(True)
    fig.savefig(FIG_DIR / "e1_f1_comparison.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_trees(trees):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6.5))
    for ax, key in zip(axes, [("구조", 2), ("all", 2)]):
        tree, cols = trees[key]
        plot_tree(tree, feature_names=cols, class_names=["합법", "불법"], filled=True,
                  impurity=False, proportion=True, rounded=True, fontsize=9, ax=ax)
        ax.set_title(f"깊이 2 결정 트리 ({key[0]} 피처)", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e1_decision_trees.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
