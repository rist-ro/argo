import numpy as np
from .SingleKMeansSim import SingleKMeansSim
from .utils import cos_similarity, riemannian_norm

class KMeansSim:
    """K-Means clustering. Uses a similarity instead of distance function."""

    def __init__(self, n_clusters, g_matrix, n_init=10, conv=1e-15):
        self._n_clusters = n_clusters
        self._n_init = n_init
        self._convergence_tol = conv
        self._g_matrix = g_matrix
        self._singlekmeans = SingleKMeansSim(n_clusters=n_clusters, g_matrix=g_matrix)
        self._reset_clusters()

    def _reset_clusters(self):
        self.score = -np.inf
        self.clusters = [[]]*self._n_clusters
        self.centers = []
        self.labels_ = []

    def _set_clusters(self, score, clusters, centers, labels_):
        self.score = score
        self.clusters = clusters
        self.centers = centers
        self.labels_ = labels_

    def _vectors_check(self, vectors, norm_vecs):
        vectors = np.array(vectors)

        assert len(vectors) >= self._n_clusters

        if norm_vecs:
            norm = riemannian_norm(vectors, self._g_matrix)
            vectors = vectors / norm.reshape(-1,1)

        # unstack vectors
        *vectors, = vectors

        return vectors

    def fit(self, vectors, norm_vecs=True):
        """Perform clustering."""
        vectors = self._vectors_check(vectors, norm_vecs)
        self.vectors = vectors
        self._reset_clusters()

        self._fit_loop()

    def _fit_loop(self):
        for i in range(self._n_init):
            clusters, centers, labels_, score = self._singlekmeans.fit(self.vectors)

            if score > self.score:
                self._set_clusters(score, clusters, centers, labels_)

    def predict(self, vectors):
        sims = cos_similarity(vectors, self.centers, g_matrix=self._g_matrix)
        labels_ = np.argmax(sims, axis=1)
        return labels_



