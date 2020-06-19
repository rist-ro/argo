import random
import numpy as np
from .utils import cos_similarity, sim_score, dist_on_sphere

class SingleKMeansSim:

    def __init__(self, n_clusters, g_matrix):
        self._n_clusters = n_clusters
        self.clusters = [[] for i in range(n_clusters)]
        self._convergence_tol = 1e-10
        self._g_matrix = g_matrix

    def _update_clusters(self):
        """Determine which cluster center each `self.vector` is closest to."""
        self.clusters = [[] for c in self.centers]

        sims = cos_similarity(self.vectors, self.centers, g_matrix=self._g_matrix)
        self.labels_ = np.argmax(sims, axis=1)

        for index, vector in zip(self.labels_, self.vectors):
            self.clusters[index].append(vector)

    def _update_centers(self):
        """Move `self.centers` to the centers of `self.clusters`.

        Return True if centers moved, else False.

        """

        new_centers = [np.mean(cl, axis=0) for cl in self.clusters]

        if np.allclose(new_centers, self.centers, atol=self._convergence_tol):
            return False

        self.centers = new_centers
        return True

    def fit(self, vectors, init='kmeans++'):
        """Perform k-means clustering."""

        self.vectors = vectors

        if init=='kmeans++':
            self.centers = smart_initialize(vectors, self._n_clusters, g_matrix=self._g_matrix)
        elif init=='random':
            self.centers = random.sample(vectors, self._n_clusters)
        elif isinstance(init, np.ndarray) or isinstance(init, list):
            self.centers = np.array(init)
            np.testing.assert_equal(self.centers.shape[0], self._n_clusters, "initial centers must be same number of the desired clusters")
        else:
            raise ValueError("init can be `kmeans++`, `random` or a list/numpy.ndarray, found: {:}".format(type(init)))

        self._update_clusters()
        while self._update_centers():
            self._update_clusters()

        self.score = sim_score(self.clusters, self.centers, g_matrix=self._g_matrix)

        return self.clusters, self.centers, self.labels_, self.score


def smart_initialize(vectors, K, g_matrix):
    C = random.sample(vectors, 1)

    for k in range(1, K):
        min_dist = np.min(dist_on_sphere(vectors, C, g_matrix=g_matrix), axis=1)
        probs = min_dist / min_dist.sum()
        indices = np.arange(len(vectors))
        i_c = np.random.choice(indices, p=probs)
        C.append(vectors[i_c])

    return C