import numpy as np

class LRU_KNN_ANNOY:
    def __init__(self, config):
        self.config = config
        self.capacity = self.config.knn_dict_capacity
        self.key_dim = self.config.knn_key_dim
        self.curr_capacity = 0
        self.delta = self.config.knn_dict_delta
        self.alpha = self.config.knn_dict_alpha

        self.embs = np.zeros((self.capacity, self.key_dim))
        self.values = np.zeros(self.capacity)
        self.terminal = np.zeros(self.capacity)
        self.embs_next = np.zeros((self.capacity, self.key_dim))

        self.lru = np.zeros(self.capacity)
        self.tm = 0.0

    def _nn(self, keys, k):
        pass

    def _nn_batch(self, keys, k):
        pass

    def queryable(self, k):
        return self.curr_capacity > k

    def query_(self, keys, k):
        assert np.ndim(keys) == 2
        keys_shape = np.shape(keys)
        embs = np.empty(keys_shape)
        vals = np.zeros(keys_shape[0])
        ters = np.zeros(keys_shape[0])
        embs_next = np.empty(keys_shape)

        if self.queryable(k):
            dists, indices = self._nn(keys, k)
            for i, ind in enumerate(indices):
                ind = ind[0]
                self.lru[ind] = self.tm
                embs[i-1] = self.embs[ind]
                vals[i-1] = self.values[ind]
                ters[i-1] = self.terminal[ind]
                embs_next[i-1] = self.embs_next[ind]
            self.tm += 0.01
        return embs, vals, ters, embs_next


    def query(self, keys, k):
        # print("np.ndim(keys): ", np.ndim(keys))
        assert np.ndim(keys) == 2
        embs = []
        vals = []
        ters = []
        embs_next = []

        if self.queryable(k):
            dists, indices = self._nn(keys, k)
            # print("indices: ", indices)
            # print("dists: ", dists)
            for ind in indices:
                self.lru[ind] = self.tm
                embs.append(self.embs[ind])
                vals.append(self.values[ind])
                ters.append(self.terminal[ind])
                embs_next.append(self.embs_next[ind])
            self.tm += 0.01
        embs = np.array(embs)
        vals = np.array(vals)
        ters = np.array(ters)
        embs_next = np.array(embs_next)
        return embs, vals, ters, embs_next

    def add(self, keys, values, terminals, keys_next):
        indices = []
        embs = []
        vals = []
        ters = []
        embs_next = []

        for i, _ in enumerate(keys):
            if self.curr_capacity >= self.capacity:
                index = np.argmin(self.lru)
            else:
                index = self.curr_capacity
                self.curr_capacity += 1
            self.lru[index] = self.tm
            indices.append(index)
            embs.append(keys[i])
            ters.append(terminals[i])
            vals.append(values[i])
            embs_next.append(keys_next[i])
        self.tm += 0.01
        self._insert(embs, vals, ters, embs_next, indices)

    def save(self, filename):
        save_data = [self.embs,
                     self.values,
                     self.terminal,
                     self.embs_next,
                     self.curr_capacity,
                     self.lru,
                     self.tm]
        np.save(filename, save_data)

    def load(self, filename):
        save_data = np.load(filename)
        self.embs, self.values, self.terminal, self.embs_next, self.curr_capacity,\
            self.lru, self.tm = save_data
        self._rebuild_index()
