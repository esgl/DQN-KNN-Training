from annoy import AnnoyIndex
import numpy as np

from knn.lru_knn_annoy import LRU_KNN_ANNOY

class Annoy_Dict(LRU_KNN_ANNOY):
    def __init__(self, config):
        super(Annoy_Dict, self).__init__(config)
        self.config = config
        self.key_dim = self.config.knn_key_dim

        self.index = AnnoyIndex(self.key_dim, metric='euclidean')
        self.index.set_seed(123)

        self.initial_update_size = self.config.knn_dict_update_step
        self.min_update_size = self.initial_update_size

        self.cached_embs = []
        self.cached_vals = []
        self.cached_terminals = []
        self.cached_embs_next = []
        self.cached_indices = []

        self.build_capacity = 0

    def _nn(self, keys, k):
        assert np.ndim(keys) == 2
        dists = []
        inds = []
        for key in keys:
            ind, dist = self.index.get_nns_by_vector(key, k, include_distances=True)
            dists.append(dist)
            inds.append(ind)
        return dists, inds

    def _insert(self, keys, values, terminal, keys_next, indices):
        self.cached_embs = self.cached_embs + keys
        self.cached_vals = self.cached_vals + values
        self.cached_terminals = self.cached_terminals + terminal
        self.cached_embs_next = self.cached_embs_next + keys_next
        self.cached_indices = self.cached_indices + indices

        if len(self.cached_indices) >= self.min_update_size:
            # self.min_update_size = max(self.initial_update_size, self.curr_capacity * 0.02)
            self._update_index()

    def _update_index(self):
        self.index.unbuild()
        for i, ind in enumerate(self.cached_indices):
            new_emb = self.cached_embs[i]
            new_val = self.cached_vals[i]
            new_t = self.cached_terminals[i]
            new_emb_next = self.cached_embs_next[i]
            self.embs[ind] = new_emb
            self.values[ind] = new_val
            self.terminal[ind] = new_t
            self.embs_next[ind] = new_emb_next
            self.index.add_item(ind, new_emb)
        self.cached_embs = []
        self.cached_vals = []
        self.cached_terminals = []
        self.cached_embs_next = []
        self.cached_indices = []

        self.index.build(50)
        self.build_capacity = self.curr_capacity

    def _rebuild(self):
        self.index.unbuild()
        for ind, emb in enumerate(self.embs[:self.curr_capacity]):
            self.index.add_item(ind, emb)
        self.index.build(50)
        self.build_capacity = self.curr_capacity

    def queryable(self, k):
        return (LRU_KNN_ANNOY.queryable(self, k) and (self.build_capacity > k))


    @property
    def capacity_(self):
        # print("self.index.get_n_items: ", self.index.get_n_items())
        return self.index.get_n_items()
