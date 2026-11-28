"""E1: 규칙 기반 베이스라인.

사람이 만든 규칙(도메인 지식)과 데이터로 찾은 단순 규칙(얕은 결정 트리)이
머신러닝(LightGBM, E2·E3)과 비교해 얼마나 잡는지 본다. 두 평가 설정으로 나눠 본다.

설정 B (거래 시점에 알 수 있는 정보만) — 피처 의미(feature_semantics.csv)를 써서 팀원 아이디어를 그대로 구현
- R1 대량 이동      : 거래 금액(local_0) >= 상위 q
- R2 분할           : 출력 수(local_4를 개수로 되돌림) >= k   (UTXO 분할)
- R2' 분할 반대     : 출력 수 <= k
- R3 다중 경유      : peel chain(입력 1개 → 출력 2개 거래가 과거 방향으로 연속) 길이 >= L
- R4 조합           : R1, R2, R3 중 2개 이상
- 결정 트리         : 과거 구조 피처 5개 / 거래 시점 피처 57개
이 규칙들은 거래 시점 정보만 쓰므로 설정 A에서도 결과가 같다.

설정 A (미래 정보 포함) — 이전 버전에서 쓴 연결 기반 대용 규칙
- R2-A 분할         : out-degree >= k   (나중에 생기는 연결)
- R2'-A 분할 반대   : out-degree <= k
- R3-A 다중 경유    : peel chain 유사 패턴(들어오는 연결 1개, 나가는 연결 1~2개) 길이 >= L
- R4-A 조합         : R3-A AND R2'-A
- 결정 트리         : 구조 피처 9개 / 전체 피처 165개

임계값은 학습 구간(1~34)에서 불법 F1이 최대가 되도록 고르고, 테스트(35~49)에 고정 적용한다.
얕은 결정 트리도 1~34로 학습하고, 확률 임계값을 같은 방식으로 학습 구간에서 고른다.

실행: python e1_rules.py   (E2·E3 결과를 참고치로 읽으므로 그 뒤에 실행)
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier, plot_tree

from common import (BASE, DARK_MARKET_STEP, RESULTS_PATH, SEED, clear_results, evaluate,
                    feature_cols, load_dataset, load_edges, log_result, split, tune_threshold,
                    z_to_count)
from graph_features import (PAST_STRUCT_PREFIX, STRUCT_PREFIX, build_graph_features,
                            build_past_graph_features)

EXPERIMENT = "E1"
FIG_DIR = BASE / "figures"
RES_DIR = BASE / "results"

CATEGORY_COLORS = {"기준": "#c3c2b7", "도메인 규칙": "#eb6834", "결정 트리": "#1baf7a",
                   "머신러닝": "#2a78d6"}
DEG_CANDS = np.arange(0, 21)
COUNT_CANDS = np.arange(1, 21)
CHAIN_CANDS = np.array([1, 2, 3, 5, 10, 20, 50, 100, 200, 500])


def best_threshold(score, y, candidates):
    """score >= t 를 불법으로 볼 때 F1이 최대인 t."""
    f1s = [f1_score(y, score >= t, zero_division=0) for t in candidates]
    return candidates[int(np.argmax(f1s))]


def period_f1(test, pred):
    early = test["time_step"].values < DARK_MARKET_STEP
    y = test["y"].values
    return (f1_score(y[early], pred[early], zero_division=0),
            f1_score(y[~early], pred[~early], zero_division=0))


def degree_table(trainval):
    rows = []
    for col, name in [("g_out_deg", "out-degree (나가는 연결)"), ("g_in_deg", "in-degree (들어오는 연결)"),
                      ("n_out", "출력 수 (local_4)"), ("n_in", "입력 수 (local_3)")]:
        bins = pd.cut(trainval[col], [-1, 0, 1, 2, 3, 5, 10, 1e9],
                      labels=["0", "1", "2", "3", "4-5", "6-10", "11+"])
        t = trainval.groupby(bins, observed=True)["y"].agg(["count", "mean"])
        t.insert(0, "feature", name)
        rows.append(t.reset_index(names="value"))
    return pd.concat(rows)


def main():
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
    clear_results(EXPERIMENT)

    df_all = load_dataset(labeled_only=False)
    edges = load_edges()
    gf = build_graph_features(df_all, edges)
    pf = build_past_graph_features(df_all, edges)
    struct_cols = [c for c in gf.columns if c.startswith(STRUCT_PREFIX)]
    pstruct_cols = [c for c in pf.columns if c.startswith(PAST_STRUCT_PREFIX)]
    df_all["n_in"] = z_to_count(df_all["local_3"].values)
    df_all["n_out"] = z_to_count(df_all["local_4"].values)
    df = (df_all[df_all["y"].notna()]
          .merge(gf[["txId"] + struct_cols], on="txId")
          .merge(pf[["txId"] + pstruct_cols], on="txId"))
    df["y"] = df["y"].astype(int)
    train, val, test = split(df)
    trainval = pd.concat([train, val])
    ytv, yte = trainval["y"].values, test["y"].values

    degree_table(trainval).to_csv(RES_DIR / "e1_degree_illicit_rate.csv", index=False,
                                  encoding="utf-8-sig")

    rows = []

    def record(setting, name, category, rule, pred, score=None, feature_set="rule"):
        metrics = evaluate(yte, pred.astype(float))
        metrics["f1"] = f1_score(yte, pred, zero_division=0)
        if score is not None:
            metrics["pr_auc"] = evaluate(yte, score)["pr_auc"]
        f1_early, f1_late = period_f1(test, pred)
        log_result(EXPERIMENT, name, feature_set, SEED, metrics, f"setting={setting}; {rule}")
        rows.append({"setting": setting, "name": name, "category": category, "rule": rule,
                     **{k: metrics[k] for k in ["precision", "recall", "f1", "pr_auc"]},
                     "f1_35_42": f1_early, "f1_43_49": f1_late, "flag_rate": pred.mean()})

    def rule_ge(col, cands):
        t = best_threshold(trainval[col].values, ytv, cands)
        return t, test[col].values >= t

    def rule_le(col, cands):
        t = -best_threshold(-trainval[col].values, ytv, -cands)
        return t, test[col].values <= t

    for setting in ["A", "B"]:
        record(setting, "모두 불법", "기준", "모든 거래 = 불법", np.ones_like(yte, dtype=bool))

    # ---- 설정 B: 거래 시점 정보만 쓰는 규칙 (피처 의미 사용) ----
    vol_cands = np.quantile(trainval["local_0"], np.linspace(0.5, 0.995, 60))
    q1, p1 = rule_ge("local_0", vol_cands)
    q1_pct = (trainval["local_0"] < q1).mean()
    record("B", "R1 대량 이동", "도메인 규칙", f"거래 금액(local_0) >= 상위 {1 - q1_pct:.1%}",
           p1, test["local_0"].values)
    k2, p2 = rule_ge("n_out", COUNT_CANDS[1:])
    record("B", "R2 분할", "도메인 규칙", f"출력 수 >= {k2}", p2, test["n_out"].values)
    k2r, p2r = rule_le("n_out", COUNT_CANDS)
    record("B", "R2' 분할 반대", "도메인 규칙", f"출력 수 <= {k2r}", p2r, -test["n_out"].values)
    L3, p3 = rule_ge("p_peel_chain", CHAIN_CANDS)
    record("B", "R3 다중 경유", "도메인 규칙", f"peel chain(입력1→출력2) >= {L3}", p3,
           test["p_peel_chain"].values)
    p4 = (p1.astype(int) + p2.astype(int) + p3.astype(int)) >= 2
    record("B", "R4 조합", "도메인 규칙", "R1·R2·R3 중 2개 이상", p4)

    # ---- 설정 A: 이전 버전의 연결 기반 대용 규칙 (나중에 생기는 연결 포함) ----
    k, p = rule_ge("g_out_deg", DEG_CANDS[1:])
    record("A", "R2-A 분할", "도메인 규칙", f"out-degree >= {k}", p, test["g_out_deg"].values)
    kr, pr = rule_le("g_out_deg", DEG_CANDS)
    record("A", "R2'-A 분할 반대", "도메인 규칙", f"out-degree <= {kr}", pr, -test["g_out_deg"].values)
    L, pl = rule_ge("g_peel_chain", CHAIN_CANDS)
    record("A", "R3-A 다중 경유", "도메인 규칙", f"peel chain 유사 패턴 >= {L}", pl,
           test["g_peel_chain"].values)
    out_tv, peel_tv = trainval["g_out_deg"].values, trainval["g_peel_chain"].values
    best = max(((f1_score(ytv, (peel_tv >= ll) & (out_tv <= kk), zero_division=0), kk, ll)
                for kk in DEG_CANDS for ll in CHAIN_CANDS))
    _, k4, L4 = best
    record("A", "R4-A 조합", "도메인 규칙", f"peel chain >= {L4} AND out-degree <= {k4}",
           (test["g_peel_chain"].values >= L4) & (test["g_out_deg"].values <= k4))

    # ---- 얕은 결정 트리 ----
    trees = {}
    tree_sets = [("A", "구조", struct_cols), ("A", "all", feature_cols(df, "all")),
                 ("B", "과거 구조", pstruct_cols), ("B", "time", feature_cols(df, "time"))]
    for setting, cols_name, cols in tree_sets:
        for depth in [1, 2, 3]:
            tree = DecisionTreeClassifier(max_depth=depth, random_state=SEED)
            tree.fit(trainval[cols], ytv)
            th = tune_threshold(ytv, tree.predict_proba(trainval[cols])[:, 1])
            prob = tree.predict_proba(test[cols])[:, 1]
            record(setting, f"트리({cols_name}) 깊이 {depth}", "결정 트리",
                   f"DecisionTree depth={depth}, threshold={th:.2f}", prob >= th, prob,
                   feature_set=cols_name)
            trees[(setting, cols_name, depth)] = (tree, cols)

    res = pd.DataFrame(rows)

    # 머신러닝 참고치 (E2, E3 결과에서 읽어 온다)
    ref = pd.read_csv(RESULTS_PATH, encoding="utf-8-sig")
    refs = [("A", "E2", "all", "LightGBM (E2, all)"),
            ("A", "E3", "all+struct+nbr+nbr2", "LightGBM (E3, 최고)"),
            ("B", "E2", "time", "LightGBM (E2, time)"),
            ("B", "E3", "time+pstruct+pnbr", "LightGBM (E3, 최고)")]
    for setting, exp, fs, label in refs:
        r = ref[(ref["experiment"] == exp) & (ref["model"] == "LightGBM") & (ref["feature_set"] == fs)
                & ref["note"].str.contains("threshold=0.5", na=False) & ~ref["note"].str.contains("tuned", na=False)]
        if len(r):
            res = pd.concat([res, pd.DataFrame([{
                "setting": setting, "name": label, "category": "머신러닝", "rule": "참고치",
                **r[["precision", "recall", "f1", "pr_auc"]].mean().to_dict()}])], ignore_index=True)

    res.to_csv(RES_DIR / "e1_summary.csv", index=False, encoding="utf-8-sig")
    plot_comparison(res)
    plot_trees(trees)

    print("\n=== E1 테스트 결과 (35~49, 임계값은 1~34에서 선택) ===")
    cols = ["setting", "name", "rule", "precision", "recall", "f1", "pr_auc", "f1_35_42", "f1_43_49",
            "flag_rate"]
    with pd.option_context("display.width", 240, "display.max_columns", 20,
                           "display.max_colwidth", 42):
        print(res[cols].round(3).to_string(index=False))
    print("\n학습 구간(1~34) 값별 불법 비율")
    print(degree_table(trainval).round(3).to_string(index=False))


def plot_comparison(res):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), sharex=True)
    for ax, setting, title in [(axes[0], "A", "[설정 A] 전체 정보 (미래 포함)"),
                               (axes[1], "B", "[설정 B] 거래 시점 정보만")]:
        part = res[res["setting"] == setting].iloc[::-1]
        ax.barh(part["name"], part["f1"], color=[CATEGORY_COLORS[c] for c in part["category"]],
                height=0.65)
        for i, v in enumerate(part["f1"]):
            ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_xlabel("테스트 불법 F1 (35~49)")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#e6e5e0")
        ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CATEGORY_COLORS.values()]
    axes[1].legend(handles, CATEGORY_COLORS.keys(), loc="upper right", frameon=False)
    fig.suptitle("규칙 기반 vs 결정 트리 vs 머신러닝", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e1_f1_comparison.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_trees(trees):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6.5))
    for ax, key in zip(axes, [("A", "all", 2), ("B", "time", 2)]):
        tree, cols = trees[key]
        plot_tree(tree, feature_names=cols, class_names=["합법", "불법"], filled=True,
                  impurity=False, proportion=True, rounded=True, fontsize=9, ax=ax)
        ax.set_title(f"[설정 {key[0]}] 깊이 2 결정 트리 ({key[1]} 피처)", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e1_decision_trees.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
