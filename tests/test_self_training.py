"""The hand-written self-training loops must match sklearn's SelfTrainingClassifier (failure record R10).

The stand-in model becomes confident about a further slice of the unlabelled rows after every fit,
so all five rounds add pseudo labels and a missing last refit changes the prediction.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.semi_supervised import SelfTrainingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import e7c_search as e7c
import e7d_test_time_st as e7d


class Growing(ClassifierMixin, BaseEstimator):
    """Confident on rows whose rank (column 0) is below training size + 20; parity (column 1) is the class."""

    def fit(self, X, y, sample_weight=None):
        self.n_ = len(y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        sure = X[:, 0] < self.n_ + 20
        p = np.where(sure, np.where(X[:, 1] > 0, 0.99, 0.01), 0.5)
        p = np.where(X[:, 0] >= 1e6, 1 / (1 + 1000 / self.n_), p)  # evaluation rows expose the training size
        return np.column_stack([1 - p, p])


def _data():
    parity = np.arange(320) % 2
    X = np.column_stack([np.arange(320, dtype=float), parity])
    Xe = np.column_stack([np.full(10, 2e6), np.zeros(10)])
    return X[:20], parity[:20].astype(int), X[20:], Xe


def _sklearn(XL, yL, XU, Xe):
    st = SelfTrainingClassifier(Growing(), threshold=0.95, max_iter=5)
    st.fit(np.vstack([XL, XU]), np.r_[yL, -np.ones(len(XU), int)])
    assert st.termination_condition_ == "max_iter"  # otherwise the test cannot see a missing refit
    return st.predict_proba(Xe)[:, 1]


def test_e7d_loop_matches_sklearn(monkeypatch):
    XL, yL, XU, Xe = _data()
    monkeypatch.setattr(e7d.e7c, "lgbm", lambda *a, **k: Growing())
    ours = e7d.self_train(XL, yL, XU, seed=0).predict_proba(Xe)[:, 1]
    np.testing.assert_allclose(ours, _sklearn(XL, yL, XU, Xe))


def test_e7c_loop_matches_sklearn(monkeypatch):
    XL, yL, XU, Xe = _data()
    monkeypatch.setattr(e7c, "lgbm", lambda *a, **k: Growing())

    class Fake:
        X = np.vstack([XL, XU, Xe])
        t = np.r_[np.ones(320, int), np.full(10, 2)]
        y = np.r_[yL, np.full(300, np.nan), np.zeros(10)]

        def split(self, fit_steps, eval_steps):
            return np.arange(320), np.arange(20), np.arange(20, 320), np.arange(320, 330)

    cfg = dict(kind="st", base="team", half_life=None, unsup=False, thr=0.95, balanced=False, plw=1.0)
    ours = e7c.run(cfg, Fake(), ("toy", [1], [2]), 1.0, 0)[0]
    np.testing.assert_allclose(ours, _sklearn(XL, yL, XU, Xe))
