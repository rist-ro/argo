import numpy as np
from .KMeansSim import KMeansSim
from .SingleKMeansSim import SingleKMeansSim
from .utils import aggregate_dissimilarity

class RepeatedBisectionSim(KMeansSim):

    def __init__(self, n_clusters, g_matrix, n_init=10, conv=1e-15, bm='agg'):
        super().__init__(n_clusters, g_matrix, n_init=n_init, conv=conv)
        self._singlebisect = SingleKMeansSim(n_clusters=2, g_matrix=g_matrix)
        bisection_measures = ['agg', 'size'] # aggregation dissimilarity or size
        if bm not in bisection_measures:
            raise ValueError("bm must be in: {:}, found {:}".format(bisection_measures, bm))
        self._bm = bm

    def _fit_loop(self):
        for i in range(self._n_init):
            rb_clusters, rb_centers = self._repeated_bisection(self.vectors)

            clusters, centers, labels_, score = self._singlekmeans.fit(self.vectors, init=rb_centers)

            if score > self.score:
                self._set_clusters(score, clusters, centers, labels_)

    def _repeated_bisection(self, vectors):
        all_clusters, all_centers, all_labels_, _ = self._singlebisect.fit(vectors)

        for i in range(self._n_clusters-2):
            next_c = self._choose_next_cluster(all_clusters, all_centers)
            choosen_cluster = all_clusters.pop(next_c)
            _ = all_centers.pop(next_c)

            b_clusters, b_centers, b_labels_, _ = self._singlebisect.fit(choosen_cluster)
            all_clusters += b_clusters
            all_centers += b_centers

        return all_clusters, all_centers

    def _choose_next_cluster(self, clusters, centers):
        if self._bm=='agg':
            measure = aggregate_dissimilarity(clusters, centers, self._g_matrix)
        elif self._bm=='size':
            measure = [len(cl) for cl in clusters]
        else:
            raise Exception('bm not understood')

        return np.argmax(measure)
