"""Anomaly detectors for E6b improved unsupervised experiment.

Implements model adapters matching team conventions:
a) GMM (full covariance, k in {8,12,16,20}, PCA {20, 0.95}) with two scorers:
   - "gmm_density" (-score_samples)
   - "gmm_component" (0.5 * logdet(Sigma_c))
b) HDBSCAN on PCA {10, 20} (subsample 20k, nearest clustered neighbor dist * (1 + label_noise))
c) Isolation Forest (n_estimators 300, max_samples {1024,4096}, max_features {0.5,1.0})
d) PCA reconstruction error (squared reconstruction error)
e) Kernel PCA reconstruction error (rbf, gamma {0.01,0.05}, 5k subsample)
f) One-class SVM (rbf, nu {0.05,0.2}, gamma {"scale",0.01}, 10k subsample of PCA-20)
g) BIRCH (threshold {0.5,1.0}, n_clusters {8,16} on PCA-20, distance to nearest centroid)
"""
from typing import Dict, Optional, Tuple, Union

import numpy as np
from sklearn.cluster import Birch, HDBSCAN, KMeans
from sklearn.decomposition import KernelPCA, PCA
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM


class GMMDetector:
    """Gaussian Mixture Model with full covariance, supporting density and component scorers."""

    def __init__(self, n_components: int = 8, pca: Optional[Union[int, float]] = 20, seed: int = 0):
        self.n_components = n_components
        self.pca = pca
        self.seed = seed
        self.pca_model = None
        self.gmm = None
        self.component_logdets_ = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        if self.pca is not None:
            if isinstance(self.pca, float):
                self.pca_model = PCA(n_components=self.pca, svd_solver="full", random_state=self.seed).fit(X)
            else:
                n_c = min(int(self.pca), X.shape[1], X.shape[0])
                self.pca_model = PCA(n_components=n_c, random_state=self.seed).fit(X)
            Z = self.pca_model.transform(X)
        else:
            self.pca_model, Z = None, X

        n_comp = min(int(self.n_components), len(Z))
        Z64 = Z.astype(np.float64)
        gmm_model = None
        for reg in [1e-4, 1e-3, 1e-2, 1e-1]:
            try:
                gmm_model = GaussianMixture(
                    n_components=n_comp,
                    covariance_type="full",
                    random_state=self.seed,
                    reg_covar=reg,
                    max_iter=100,
                ).fit(Z64)
                break
            except (ValueError, np.linalg.LinAlgError):
                continue
        if gmm_model is None:
            # Fallback with tied covariance if full fails on extreme degeneracies
            gmm_model = GaussianMixture(
                n_components=n_comp,
                covariance_type="tied",
                random_state=self.seed,
                reg_covar=1e-2,
                max_iter=100,
            ).fit(Z64)
        self.gmm = gmm_model

        # Compute 0.5 * logdet(Sigma_c) for all components
        covs = self.gmm.covariances_
        logdets = []
        if self.gmm.covariance_type == "full":
            for c in range(len(covs)):
                sign, ldet = np.linalg.slogdet(covs[c])
                logdets.append(float(0.5 * ldet if sign > 0 else -100.0))
        else:
            sign, ldet = np.linalg.slogdet(covs)
            val = float(0.5 * ldet if sign > 0 else -100.0)
            logdets = [val] * n_comp
        self.component_logdets_ = np.array(logdets, dtype=np.float32)
        return self

    def score_density(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score: negative log-likelihood (higher = more anomalous)."""
        X = np.asarray(X, dtype=np.float32)
        Z = self.pca_model.transform(X) if self.pca_model is not None else X
        return (-self.gmm.score_samples(Z.astype(np.float64))).astype(np.float32)

    def score_component(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score: 0.5 * logdet(Sigma_c) of argmax component (denser = lower)."""
        X = np.asarray(X, dtype=np.float32)
        Z = self.pca_model.transform(X) if self.pca_model is not None else X
        c_pred = self.gmm.predict(Z.astype(np.float64))
        return self.component_logdets_[c_pred].astype(np.float32)


class HDBSCANDetector:
    """HDBSCAN cluster distance anomaly detector on PCA space."""

    def __init__(
        self,
        pca: int = 10,
        min_cluster_size: int = 25,
        min_samples: int = 5,
        subsample_size: int = 20000,
        seed: int = 0,
    ):
        self.pca = pca
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.subsample_size = subsample_size
        self.seed = seed
        self.pca_model = None
        self.hdb = None
        self.nn_ref = None
        self.ref_noise = None
        self.cluster_sizes_ = {}

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        rng = np.random.RandomState(self.seed)

        if self.pca is not None:
            n_c = min(int(self.pca), X.shape[1], X.shape[0])
            self.pca_model = PCA(n_components=n_c, random_state=self.seed).fit(X)
            Z = self.pca_model.transform(X)
        else:
            self.pca_model, Z = None, X

        # Subsample if necessary
        if len(Z) > self.subsample_size:
            idx = np.sort(rng.choice(len(Z), self.subsample_size, replace=False))
            Z_sub = Z[idx]
        else:
            Z_sub = Z

        mcs = min(int(self.min_cluster_size), max(2, len(Z_sub) // 2))
        ms = min(int(self.min_samples), mcs)

        self.hdb = HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            copy=True,
        ).fit(Z_sub)

        labels = self.hdb.labels_
        unique_labels = [c for c in np.unique(labels) if c >= 0]
        self.cluster_sizes_ = {int(c): int(np.sum(labels == c)) for c in unique_labels}

        clustered_mask = (labels >= 0)
        if np.any(clustered_mask):
            Z_ref = Z_sub[clustered_mask]
            probs = self.hdb.probabilities_[clustered_mask]
            ref_noise = (1.0 - probs).astype(np.float32)
        else:
            Z_ref = Z_sub
            ref_noise = np.ones(len(Z_sub), dtype=np.float32)

        self.nn_ref = NearestNeighbors(n_neighbors=1).fit(Z_ref)
        self.ref_noise = ref_noise
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        Z = self.pca_model.transform(X) if self.pca_model is not None else X
        dists, indices = self.nn_ref.kneighbors(Z, return_distance=True)
        dists = dists[:, 0].astype(np.float32)
        nearest_idx = indices[:, 0]
        noise_factor = 1.0 + self.ref_noise[nearest_idx]
        return (dists * noise_factor).astype(np.float32)


class IForestDetector:
    """Isolation Forest anomaly detector."""

    def __init__(
        self,
        n_estimators: int = 300,
        max_samples: Union[int, float] = 1024,
        max_features: float = 1.0,
        seed: int = 0,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.seed = seed
        self.iforest = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        ms = min(int(self.max_samples), len(X)) if isinstance(self.max_samples, int) else self.max_samples
        self.iforest = IsolationForest(
            n_estimators=int(self.n_estimators),
            max_samples=ms,
            max_features=float(self.max_features),
            random_state=self.seed,
            n_jobs=-1,
        ).fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return (-self.iforest.score_samples(X)).astype(np.float32)


class PCAReconDetector:
    """PCA reconstruction error anomaly detector."""

    def __init__(self, n_components: int = 5, seed: int = 0):
        self.n_components = n_components
        self.seed = seed
        self.pca = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        n_c = min(int(self.n_components), X.shape[1], X.shape[0])
        self.pca = PCA(n_components=n_c, random_state=self.seed).fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        z = self.pca.transform(X)
        x_recon = self.pca.inverse_transform(z)
        return np.sum((X - x_recon) ** 2, axis=1, dtype=np.float32)


class KPCAReconDetector:
    """Kernel PCA reconstruction error anomaly detector."""

    def __init__(
        self,
        n_components: int = 20,
        gamma: float = 0.01,
        subsample_size: int = 5000,
        seed: int = 0,
    ):
        self.n_components = n_components
        self.gamma = gamma
        self.subsample_size = subsample_size
        self.seed = seed
        self.kpca = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        rng = np.random.RandomState(self.seed)
        if len(X) > self.subsample_size:
            idx = np.sort(rng.choice(len(X), self.subsample_size, replace=False))
            X_sub = X[idx]
        else:
            X_sub = X

        n_c = min(int(self.n_components), X_sub.shape[1], X_sub.shape[0])
        self.kpca = KernelPCA(
            n_components=n_c,
            kernel="rbf",
            gamma=float(self.gamma),
            fit_inverse_transform=True,
            random_state=self.seed,
        ).fit(X_sub)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        z = self.kpca.transform(X)
        x_recon = self.kpca.inverse_transform(z)
        return np.sum((X - x_recon) ** 2, axis=1, dtype=np.float32)


class OCSVMDetector:
    """One-Class SVM anomaly detector on PCA-20 representation."""

    def __init__(
        self,
        nu: float = 0.05,
        gamma: Union[str, float] = "scale",
        pca: int = 20,
        subsample_size: int = 10000,
        seed: int = 0,
    ):
        self.nu = nu
        self.gamma = gamma
        self.pca = pca
        self.subsample_size = subsample_size
        self.seed = seed
        self.pca_model = None
        self.ocsvm = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        if self.pca is not None:
            n_c = min(int(self.pca), X.shape[1], X.shape[0])
            self.pca_model = PCA(n_components=n_c, random_state=self.seed).fit(X)
            Z = self.pca_model.transform(X)
        else:
            self.pca_model, Z = None, X

        rng = np.random.RandomState(self.seed)
        if len(Z) > self.subsample_size:
            idx = np.sort(rng.choice(len(Z), self.subsample_size, replace=False))
            Z_sub = Z[idx]
        else:
            Z_sub = Z

        gam = float(self.gamma) if self.gamma != "scale" else "scale"
        self.ocsvm = OneClassSVM(
            kernel="rbf",
            nu=float(self.nu),
            gamma=gam,
        ).fit(Z_sub)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        Z = self.pca_model.transform(X) if self.pca_model is not None else X
        return (-self.ocsvm.decision_function(Z)).astype(np.float32)


class BirchDetector:
    """BIRCH clustering distance to nearest subcluster centroid on PCA-20."""

    def __init__(
        self,
        threshold: float = 0.5,
        n_clusters: int = 8,
        pca: int = 20,
        seed: int = 0,
        subsample_size: int = 10000,
    ):
        self.subsample_size = subsample_size
        self.threshold = threshold
        self.n_clusters = n_clusters
        self.pca = pca
        self.seed = seed
        self.pca_model = None
        self.birch = None
        self.nn = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        if self.pca is not None:
            n_c = min(int(self.pca), X.shape[1], X.shape[0])
            self.pca_model = PCA(n_components=n_c, random_state=self.seed).fit(X)
            Z = self.pca_model.transform(X)
        else:
            self.pca_model, Z = None, X

        # Fit on a subsample like the other Stage 2 models. Birch's own global step
        # (AgglomerativeClustering over all subclusters) is O(n_subclusters^2) memory and killed
        # the Colab runtime, so build subclusters only and group them with a size-weighted K-Means.
        rng = np.random.RandomState(self.seed)
        if len(Z) > self.subsample_size:
            Z = Z[np.sort(rng.choice(len(Z), self.subsample_size, replace=False))]
        self.birch = Birch(threshold=float(self.threshold), n_clusters=None).fit(Z)
        sub = self.birch.subcluster_centers_
        weights = np.bincount(self.birch.labels_, minlength=len(sub)).astype(float)
        n_cl = min(int(self.n_clusters), len(sub))
        km = KMeans(n_clusters=n_cl, n_init=10, random_state=self.seed).fit(sub, sample_weight=weights)
        self.n_subclusters_ = int(len(sub))
        self.nn = NearestNeighbors(n_neighbors=1).fit(km.cluster_centers_)
        # The CF-tree (linked leaves) is not needed for scoring and its depth makes
        # joblib/pickle hit RecursionError with thousands of subclusters; drop it.
        self.birch = None
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        Z = self.pca_model.transform(X) if self.pca_model is not None else X
        dists, _ = self.nn.kneighbors(Z, return_distance=True)
        return dists[:, 0].astype(np.float32)


def make_detector(model_type: str, params: Optional[Dict] = None, seed: int = 0):
    """Instantiate a detector from model_type and parameters."""
    p = dict(params or {})
    m = model_type.lower()
    if m in ("gmm", "gmm_density", "gmm_component"):
        return GMMDetector(seed=seed, **p)
    elif m == "hdbscan":
        return HDBSCANDetector(seed=seed, **p)
    elif m in ("iforest", "isolation_forest"):
        return IForestDetector(seed=seed, **p)
    elif m == "pca_recon":
        return PCAReconDetector(seed=seed, **p)
    elif m == "kpca_recon":
        return KPCAReconDetector(seed=seed, **p)
    elif m == "ocsvm":
        return OCSVMDetector(seed=seed, **p)
    elif m == "birch":
        return BirchDetector(seed=seed, **p)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
