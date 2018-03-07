import numpy as np

class LRU_KNN_NMSlib:
    def __init__(self, config):
        self.capacity = config.knn_dict_capacity
        self.emb_dim = config.knn_key_dim
        self.current_capacity = 0
        self.delta = config.knn_dict_delta
        self.alpha = config.knn_dict_alpha

        self.embs = np.zeros((self.capacity, self.emb_dim), dtype=np.float16)
        self.values = np.zeros(self.capacity, dtype=np.uint8)
        self.embs_next = np.zeros((self.capacity, self.emb_dim), dtype=np.float16)
        self.lru = np.zeros(self.capacity, dtype=np.float32)
        self.indices = np.zeros(self.capacity, dtype=np.int64)
        self.tm = 0.0

    def _nn(self, keys, k):
        pass

    def _nn_batch(self, keys, k):
        pass

    def _insert(self, embs, values, embs_next, indices):
        pass

    def queryable(self, k):
        return self.current_capacity > k

    def query_batch(self, embs, k):
        assert np.ndim(embs) == 2
        if self.queryable(k):
            indices, _ = self._nn_batch(keys=embs, k=k)
            embs = []
            values = []
            embs_next = []
            for i in range(indices.shape[0]):
                indices_ = indices[i]
                embs_ = []
                values_ = []
                embs_next_ = []
                for ind in indices_:
                    self.lru[ind] = self.tm
                    embs_.append(self.embs[ind])
                    values_.append(self.values[ind])
                    embs_next_.append(self.embs_next[ind])
                embs.append(embs_)
                values.append(values_)
                embs_next.append(embs_next_)
            self.tm += 0.01
        return embs, values, embs_next

    def query(self, embs, k):
        assert np.ndim(embs) == 1
        indices, _ = self._nn(keys=embs, k=k)

        embs = []
        values = []
        embs_next = []

        for ind in indices:
            self.lru[ind] = self.tm
            embs.append(self.embs[ind])
            values.append(self.values[ind])
            embs_next.append(self.embs_next)
        self.tm += 0.01
        return embs, values, embs_next

    def add(self, embs, values, embs_next):
        indices, embs_, values_, embs_next_ = [], [], [], []

        for i, _ in enumerate(embs):
            if self.current_capacity >= self.capacity:
                index = np.argmin(self.lru)
            else:
                index = self.current_capacity
                self.current_capacity += 1
            self.lru[index] = self.tm
            indices.append(index)
            embs_.append(embs[i])
            values_.append(values[i])
            embs_next_.append(embs_next[i])
        self._insert(embs_, values_, embs_next_, indices)

        self.tm += 0.01

    def save(self, filename):
        save_data = [self.embs,
                     self.values,
                     self.embs_next,
                     self.current_capacity,
                     self.lru,
                     self.tm]
        np.save(filename, save_data)

    def load(self, filename):
        save_data = np.load(filename)
        self.embs, self.values, self.embs_next, self.current_capacity, self.lru, self.tm = save_data
        self._rebuild_index()