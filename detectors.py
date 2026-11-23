"""Small, label-free anomaly detector adapters with a common score direction."""
import numpy as np
from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler


class Detector:
    def __init__(self, name, params, seed=0):
        self.name, self.params, self.seed = name, dict(params), seed
        self.pca = None
        self.scaler = None

    def _transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        X = self.scaler.transform(X) if self.scaler is not None else X
        return self.pca.transform(X) if self.pca is not None else X

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        if self.name == "dbscan" and len(X) > 10000:
            rng = np.random.RandomState(self.seed)
            X = X[np.sort(rng.choice(len(X), 10000, replace=False))]
        elif self.name == "lof" and len(X) > 50000:
            rng = np.random.RandomState(self.seed)
            X = X[np.sort(rng.choice(len(X), 50000, replace=False))]

        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        pca = self.params.get("pca")
        if pca:
            self.pca = PCA(n_components=int(pca), random_state=self.seed).fit(Xs)
            Z = self.pca.transform(Xs)
        else:
            self.pca, Z = None, Xs
        self._fit_X = X
        p = {k: v for k, v in self.params.items() if k != "pca"}
        if self.name == "kmeans":
            algo = p.pop("algorithm", "KMeans")
            cls = MiniBatchKMeans if algo == "MiniBatchKMeans" else KMeans
            self.model = cls(n_clusters=p.pop("n_clusters", 8), random_state=self.seed,
                             n_init=10, **p).fit(Z)
        elif self.name == "dbscan":
            quantile = p.pop("eps_quantile", None)
            mins = int(p.get("min_samples", 5))
            distances = NearestNeighbors(n_neighbors=min(mins + 1, len(Z))).fit(Z).kneighbors(Z)[0][:, -1]
            self.kdist_ = np.sort(distances)
            if quantile is not None:
                self.eps_ = float(np.quantile(self.kdist_, float(quantile)))
                p["eps"] = self.eps_
            elif "eps" in p:
                self.eps_ = float(p["eps"])
            self.model = DBSCAN(**p).fit(Z)
            self.core = Z[self.model.core_sample_indices_]
            if len(self.core) == 0:
                self.core = Z
            self.neighbors = NearestNeighbors(n_neighbors=1).fit(self.core)
        elif self.name == "gmm":
            self.model = GaussianMixture(random_state=self.seed, **p).fit(Z)
        elif self.name == "bgmm":
            self.model = BayesianGaussianMixture(random_state=self.seed, **p).fit(Z)
        elif self.name == "iforest":
            self.model = IsolationForest(random_state=self.seed, **p).fit(Z)
        elif self.name == "lof":
            self.model = LocalOutlierFactor(novelty=True, **p).fit(Z)
        else:
            raise ValueError(f"unknown detector: {self.name}")
        return self

    def score(self, X):
        Z = self._transform(X)
        if self.name == "kmeans":
            return self.model.transform(Z).min(axis=1)
        if self.name == "dbscan":
            return self.neighbors.kneighbors(Z, return_distance=True)[0][:, 0]
        if self.name in ("gmm", "bgmm", "iforest", "lof"):
            return -self.model.score_samples(Z)


def make_detector(name, params=None, seed=0):
    name = name.lower()
    if name not in {"kmeans", "dbscan", "gmm", "bgmm", "iforest", "lof"}:
        raise ValueError(f"unknown detector: {name}")
    return Detector(name, params or {}, seed=seed)
