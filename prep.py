"""Preprocessing pipelines for Elliptic unsupervised learning (E6b).

This module implements the five preprocessing pipelines (P0-P4) for experiment E6b:
- P0 "std": All 165 features, StandardScaler (old baseline).
- P1 "std_notp": 165 minus TIME_PROXY, StandardScaler.
- P2 "robust_clip": 165 minus TIME_PROXY, clip to train 0.5/99.5 percentiles, RobustScaler.
- P3 "quantile": 165 minus TIME_PROXY, QuantileTransformer(output_distribution="normal",
                 n_quantiles=1000, subsample=100000, random_state=seed).
- P3b "quantile_all" (setting B only): quantile on all setting features incl. TIME_PROXY.
- P4 "stepnorm_local": local_* only (93 features; setting B: TX local_0-20), standardize each row by the mean/std
  of ITS OWN time step computed over all (unlabeled) rows of that step, then StandardScaler
  fit on fit rows.

Transductive Per-Step Normalization Note (P4):
P4 is transductive strictly per time step. For each discrete time step, normalization
statistics (mean and standard deviation) are computed using only the transactions present
within that specific time step (including unlabeled transactions, with no labels accessed).
No information is shared across different time steps, ensuring complete causal temporal
integrity without future-to-past or past-to-future leakage across time steps.
"""
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler

TIME_PROXY_ALL = ["agg_7", "agg_9", "agg_6", "agg_42", "agg_45", "agg_43"]

# Evaluation settings (team common.py):
#   A = all 165 features (includes future info: address stats local_21-92, successor aggregates agg_36-71)
#   B = 57 features knowable at transaction time: TX (local_0-20) + NBR_IN (agg_0-35)
SETTING = "A"
TIME_PROXY = list(TIME_PROXY_ALL)
ALL_FEATURES = [f"local_{i}" for i in range(93)] + [f"agg_{i}" for i in range(72)]
NOTP_FEATURES = [f for f in ALL_FEATURES if f not in TIME_PROXY]
LOCAL_FEATURES = [f"local_{i}" for i in range(93)]

PREP_NAMES = ["std", "std_notp", "robust_clip", "quantile", "stepnorm_local"]


def configure(setting: str) -> None:
    """Switch the module-wide feature lists to evaluation setting "A" or "B".

    Call before creating preprocessors and before set_step_stats.
    Setting B adds "quantile_all" (quantile with the time-proxy features kept) so the
    effect of time-proxy removal under quantile scaling can be measured.
    """
    global SETTING, TIME_PROXY, ALL_FEATURES, NOTP_FEATURES, LOCAL_FEATURES, PREP_NAMES
    if setting == "A":
        ALL_FEATURES = [f"local_{i}" for i in range(93)] + [f"agg_{i}" for i in range(72)]
        LOCAL_FEATURES = [f"local_{i}" for i in range(93)]
        PREP_NAMES = ["std", "std_notp", "robust_clip", "quantile", "stepnorm_local"]
    elif setting == "B":
        ALL_FEATURES = [f"local_{i}" for i in range(21)] + [f"agg_{i}" for i in range(36)]
        LOCAL_FEATURES = [f"local_{i}" for i in range(21)]
        PREP_NAMES = ["std", "std_notp", "robust_clip", "quantile", "quantile_all", "stepnorm_local"]
    else:
        raise ValueError(f"unknown setting: {setting}")
    SETTING = setting
    TIME_PROXY = [f for f in TIME_PROXY_ALL if f in ALL_FEATURES]
    NOTP_FEATURES = [f for f in ALL_FEATURES if f not in TIME_PROXY]


def get_feature_names(prep_name: str) -> List[str]:
    """Return the list of input feature column names for the given preprocessing pipeline."""
    if prep_name in ("std", "quantile_all"):
        return list(ALL_FEATURES)
    if prep_name in ("std_notp", "robust_clip", "quantile"):
        return list(NOTP_FEATURES)
    if prep_name == "stepnorm_local":
        return list(LOCAL_FEATURES)
    raise ValueError(f"Unknown prep_name: {prep_name}. Expected one of {PREP_NAMES}")


_STEP_STATS = {}


def set_step_stats(df_all: pd.DataFrame) -> None:
    """Per-step mean/std of local features over ALL rows of each step (labeled + unknown).

    Uses only unlabeled feature values of each step itself (no labels, no other steps).
    Must be called once with the full dataset before using P4, so that val/test rows
    (which are labeled-only subsets) are normalized with the same per-step statistics.
    """
    local_cols = list(LOCAL_FEATURES)
    _STEP_STATS.clear()
    for s, g in df_all.groupby("time_step"):
        v = g[local_cols].to_numpy(dtype=np.float32)
        _STEP_STATS[int(s)] = (v.mean(axis=0), v.std(axis=0))


class StepwiseLocalStandardizer(BaseEstimator, TransformerMixin):
    """Standardize local features by the mean/std of each row's own time step.

    Transductive per step: uses only the unlabeled features of the transaction's
    own time step (no labels, no other steps). A global StandardScaler is then
    fit on the training/fit rows of the resulting step-normalized features.
    """

    def __init__(self):
        self.global_scaler = StandardScaler()
        self.feature_names_ = list(LOCAL_FEATURES)

    def _extract_parts(self, X):
        if isinstance(X, pd.DataFrame):
            time_step = X["time_step"].values.astype(int)
            local_cols = [c for c in X.columns if c.startswith("local_")]
            if not local_cols:
                # If local_ cols not named, drop time_step and take first 93 cols
                other_cols = [c for c in X.columns if c != "time_step"]
                vals = X[other_cols[:93]].values.astype(np.float32)
            else:
                vals = X[local_cols].values.astype(np.float32)
            return time_step, vals
        elif isinstance(X, np.ndarray):
            # Assume col 0 is time_step, remaining columns are local features
            time_step = X[:, 0].astype(int)
            vals = X[:, 1:].astype(np.float32)
            return time_step, vals
        else:
            raise TypeError(f"Unsupported input type: {type(X)}")

    def _step_normalize(self, time_step: np.ndarray, vals: np.ndarray) -> np.ndarray:
        norm_vals = np.empty_like(vals, dtype=np.float32)
        unique_steps = np.unique(time_step)
        for s in unique_steps:
            mask = (time_step == s)
            step_v = vals[mask]
            if int(s) in _STEP_STATS:  # stats of ALL rows of the step (set_step_stats)
                mean, std = _STEP_STATS[int(s)]
            else:  # fallback: stats of the passed rows only
                mean, std = np.mean(step_v, axis=0), np.std(step_v, axis=0)
            std = np.where(std == 0.0, 1.0, std)
            norm_vals[mask] = (step_v - mean) / std
        return norm_vals

    def fit(self, X, y=None):
        time_step, vals = self._extract_parts(X)
        step_normed = self._step_normalize(time_step, vals)
        self.global_scaler.fit(step_normed)
        return self

    def transform(self, X):
        time_step, vals = self._extract_parts(X)
        step_normed = self._step_normalize(time_step, vals)
        return self.global_scaler.transform(step_normed).astype(np.float32)


class RobustClipScaler(BaseEstimator, TransformerMixin):
    """Clip features to train 0.5/99.5 percentiles, then apply RobustScaler."""

    def __init__(self):
        self.scaler = RobustScaler()
        self.p_low_ = None
        self.p_high_ = None

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=np.float32)
        self.p_low_ = np.percentile(X_arr, 0.5, axis=0)
        self.p_high_ = np.percentile(X_arr, 99.5, axis=0)
        clipped = np.clip(X_arr, self.p_low_, self.p_high_)
        self.scaler.fit(clipped)
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=np.float32)
        clipped = np.clip(X_arr, self.p_low_, self.p_high_)
        return self.scaler.transform(clipped).astype(np.float32)


class EllipticPreprocessor(BaseEstimator, TransformerMixin):
    """Complete preprocessor handling P0-P4 with optional PCA reduction."""

    def __init__(self, prep_name: str = "std", pca: Optional[Union[int, float]] = None, seed: int = 0):
        if prep_name not in PREP_NAMES:
            raise ValueError(f"Unknown prep_name: {prep_name}. Expected one of {PREP_NAMES}")
        self.prep_name = prep_name
        self.pca = pca
        self.seed = seed
        self.feature_names_ = get_feature_names(prep_name)
        self.base_transformer_ = None
        self.pca_transformer_ = None

    def _select_input(self, X):
        if isinstance(X, pd.DataFrame):
            if self.prep_name == "stepnorm_local":
                # Stepwise needs time_step column
                cols = ["time_step"] + [c for c in self.feature_names_ if c in X.columns]
                return X[cols]
            else:
                cols = [c for c in self.feature_names_ if c in X.columns]
                return X[cols].values.astype(np.float32)
        elif isinstance(X, np.ndarray):
            return X
        raise TypeError(f"Unsupported input type: {type(X)}")

    def fit(self, X, y=None):
        X_in = self._select_input(X)

        if self.prep_name == "std":
            self.base_transformer_ = StandardScaler().fit(X_in)
            Z = self.base_transformer_.transform(X_in)
        elif self.prep_name == "std_notp":
            self.base_transformer_ = StandardScaler().fit(X_in)
            Z = self.base_transformer_.transform(X_in)
        elif self.prep_name == "robust_clip":
            self.base_transformer_ = RobustClipScaler().fit(X_in)
            Z = self.base_transformer_.transform(X_in)
        elif self.prep_name in ("quantile", "quantile_all"):
            n_q = min(1000, len(X_in))
            self.base_transformer_ = QuantileTransformer(
                output_distribution="normal",
                n_quantiles=n_q,
                subsample=100000,
                random_state=self.seed,
            ).fit(X_in)
            Z = self.base_transformer_.transform(X_in)
        elif self.prep_name == "stepnorm_local":
            self.base_transformer_ = StepwiseLocalStandardizer().fit(X_in)
            Z = self.base_transformer_.transform(X_in)

        if self.pca is not None:
            if isinstance(self.pca, float):
                self.pca_transformer_ = PCA(
                    n_components=self.pca,
                    svd_solver="full",
                    random_state=self.seed,
                ).fit(Z)
            else:
                n_c = min(int(self.pca), Z.shape[1], Z.shape[0])
                self.pca_transformer_ = PCA(
                    n_components=n_c,
                    random_state=self.seed,
                ).fit(Z)
        else:
            self.pca_transformer_ = None

        return self

    def transform(self, X):
        X_in = self._select_input(X)
        Z = self.base_transformer_.transform(X_in)
        if self.pca_transformer_ is not None:
            Z = self.pca_transformer_.transform(Z)
        return Z.astype(np.float32)


def make_preprocessor(prep_name: str, pca: Optional[Union[int, float]] = None, seed: int = 0) -> EllipticPreprocessor:
    """Factory creating an EllipticPreprocessor instance."""
    return EllipticPreprocessor(prep_name=prep_name, pca=pca, seed=seed)
