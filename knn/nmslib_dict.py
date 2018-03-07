import nmslib
from knn.lru_knn_nmslib import LRU_KNN_NMSlib
from multiprocessing import cpu_count
import numpy as np

class NMSlib_Dict(LRU_KNN_NMSlib):
    def __init__(self, config):
        super(NMSlib_Dict, self).__init__(config)
        self.print_progress = config.nmslib_print_progress

        self.index = nmslib.init(method="hnsw", space="cosinesimil")
        self.initial_update_size = config.knn_dict_update_step
        self.min_update_size = self.initial_update_size

        self.cached_embs = []
        self.cached_values = []
        self.cached_embs_next = []
        self.cached_indices = []

        self.build_capacity = 0

    def _nn(self, keys, k):
        assert np.ndim(keys) == 1
        nn = self.index.knnQuery(queries=keys, k=k)
        inds = np.array(nn, dtype=np.int)[0]
        dists = np.array(nn, dtype=np.float64)[0]
        return inds, dists

    def _nn_batch(self, keys, k):
        print("np.ndim(keys)", np.ndim(keys))
        assert np.ndim(keys) == 2
        print("k: ", k)
        nn = self.index.knnQueryBatch(queries=keys, k=k, num_threads=cpu_count())
        print("nn: ", nn)
        inds = np.array(nn, dtype=np.int)[:,0,:]
        print("inds: ", inds)
        dists = np.array(nn, dtype=np.float64)[:,1,:]
        return inds, dists

    def _insert(self, embs, values, embs_next, indices):
        self.cached_embs = self.cached_embs + embs
        self.cached_values = self.cached_values + values
        self.cached_embs_next = self.cached_embs_next + embs_next
        self.cached_indices = self.cached_indices + indices

        if len(self.cached_indices) > self.min_update_size:
            self.min_update_size = max(self.min_update_size, self.current_capacity * 0.02)
            self._update_index()


    def _update_index(self):
        self.index = nmslib.init(method="hnsw", space="cosinesimil")

        self.index.addDataPointBatch(data=self.embs[:self.current_capacity], ids=self.indices)
        self.index.createIndex(print_progress=self.print_progress)

    def _rebuild_index(self):
        self.index = nmslib.init(method="hnsw", space="cosinesimil")
        self.index.addDataPointBatch(data=self.embs[:self.current_capacity])
        self.index.createIndex(print_progress=self.print_progress)
