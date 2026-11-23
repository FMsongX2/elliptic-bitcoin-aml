"""The report must follow from results/ (failure records R1, R7, R8).

Every table is regenerated and compared, and each numbered conclusion is tested against the CSVs.
Prose outside the conclusions is not parsed.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import report_tables as T


@pytest.fixture(scope="module")
def text():
    try:
        p = T.report_path()
    except StopIteration:
        pytest.skip("report not shipped")
    return p.read_bytes().decode("utf-8").replace("\r\n", "\n")


@pytest.fixture(scope="module")
def ap():
    return T.rd("eval_seed_ap").groupby("method").pr_auc.mean()


@pytest.fixture(scope="module")
def paired():
    return T.rd("eval_paired").set_index(["a", "b"])


def test_every_table_is_current(text):
    assert sorted(n for n, _ in T.MARK.findall(text)) == sorted(T.TABLES)
    assert T.fill(text) == text


def test_no_tilde_or_strikethrough(text):
    assert "~" not in text and "<del>" not in text and "<s>" not in text


def test_every_figure_exists_and_is_numbered_in_order(text):
    import re
    for src in re.findall(r"\]\((\.\./figures/[^)]+)\)", text):
        assert (T.ROOT / "docs" / src).exists(), src
    nums = [int(n) for n in re.findall(r"^\*\*그림 (\d+)\.", text, re.M)]
    assert nums == list(range(1, len(nums) + 1))


def test_headline_range(text, ap):
    few = [ap[f"{m} {p}"] for m in ("LGBM", "ST") for p in ("5%", "10%")]
    assert f"PR-AUC {min(few):.2f}-{max(few):.2f}까지" in text


def test_conclusion_1_birch_best_unsupervised(text):
    s = T.rd("e6b_B_summary")
    s = s[s["mode"] == "best"].sort_values("val_pr_auc", ascending=False).drop_duplicates("model")
    best = s.loc[s.pr_auc_mean.idxmax()]
    assert best.model == "BIRCH"
    assert f"BIRCH가 {best.pr_auc_mean:.3f}다" in text


def test_conclusion_2_few_labels_vs_all_is_undecided(text, ap, paired):
    assert f"{ap['LGBM 1%']:.2f}-{ap['ST 1%']:.2f}가 된다" in text
    for p in ("5%", "10%"):
        assert T.verdict(paired.loc[(f"ST {p}", "LGBM 100%")]) == "판정 불가"


def test_conclusion_3_self_training_only_at_one_percent(text, paired):
    r = paired.loc[("ST 1%", "LGBM 1%")]
    assert T.verdict(r) == "**높다**"
    assert f"LightGBM보다 {r['diff']:.3f} ± {r.seed_se:.3f} 높고" in text
    for p in ("5%", "10%"):
        assert T.verdict(paired.loc[(f"ST {p}", f"LGBM {p}")]) == "판정 불가"


def test_conclusion_4_ensembles_not_better(paired):
    for rule in ("past", "prob"):
        for p in ("1%", "5%", "10%"):
            assert T.verdict(paired.loc[(f"ENS-{rule} {p}", f"ST {p}")]) != "**높다**"


def test_conclusion_5_shift(text):
    s = T.rd("theory_shift").set_index(["period", "group"])
    a, b = s.loc[("1-34", "single")], s.loc[("43-49", "single")]
    assert f"{a.share_of_illicit:.0%}에서 {b.share_of_illicit:.0%}로" in text
    assert f"{a.p_illicit_in_group:.1%}에서 {b.p_illicit_in_group:.1%}가" in text


def test_conclusion_6_recovery_and_no_label_fixes(text):
    a = T.rd("e8_partA")
    m = a.groupby("condition").pr_auc_46_49.mean()
    assert f"1-42 label만으로 {m['train_1_42']:.3f}다" in text
    assert f"더하면 {m['train_1_42_post0.1_w1']:.3f}, 1,046건이면 {m['train_1_42_post0.25_w1']:.3f}" in text
    assert f"전부면 {m['train_1_42_post1_w1']:.3f}이다" in text
    d = a.pivot_table(index="seed", columns="condition", values="pr_auc_46_49")
    gain = d["train_1_42_post0.1_w1"] - d["train_1_42"]
    assert gain.mean() > T.crit(len(gain)) * gain.sem()
    s = T.rd("e9_summary")
    s = s[(s.train == "1-34") & (s["eval"] == "35-49")].set_index("method").pr_auc
    assert s.idxmax() == "global"
    h = T.rd("e9_hedge").set_index(["train", "eval"]).loc[("1-34", "35-49")]
    assert max(h.hedge_past, h.hedge_prob) < s["global"]
