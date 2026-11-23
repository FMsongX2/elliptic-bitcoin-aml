"""The reported ensembles must not use any later time step (failure record R9)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_reliability as er


def _example():
    rng = np.random.RandomState(0)
    steps = np.repeat([35, 36, 37, 38], 25)
    return steps, [rng.rand(100) for _ in range(3)]


@pytest.mark.parametrize("rule", ["past", "prob"])
def test_earlier_scores_ignore_later_steps(rule):
    steps, ps = _example()
    early = steps <= 36
    changed = [p.copy() for p in ps]
    for k, p in enumerate(changed):
        p[~early] = (k + 1) / 10.0
    before = er.ensemble_scores(ps, steps, rule)
    after = er.ensemble_scores(changed, steps, rule)
    np.testing.assert_allclose(before[early], after[early])


@pytest.mark.parametrize("rule", ["past", "prob"])
def test_earlier_scores_same_without_later_rows(rule):
    steps, ps = _example()
    early = steps <= 36
    full = er.ensemble_scores(ps, steps, rule)
    cut = er.ensemble_scores([p[early] for p in ps], steps[early], rule)
    np.testing.assert_allclose(full[early], cut)


def test_whole_period_rank_is_not_offered():
    steps, ps = _example()
    with pytest.raises(ValueError):
        er.ensemble_scores(ps, steps, "global")


def test_weighted_bootstrap_ap_matches_sklearn():
    from sklearn.metrics import average_precision_score
    rng = np.random.RandomState(1)
    y = rng.randint(0, 2, 300)
    p = np.round(rng.rand(300), 2)  # ties
    idx = rng.choice(300, 300)
    w = np.bincount(idx, minlength=300)
    assert er.boot_ap(y, p, [w])[0] == pytest.approx(average_precision_score(y[idx], p[idx]))
