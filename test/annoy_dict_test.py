from knn.annoy_dict import Annoy_Dict
import numpy as np
import time
class config:
    scale = 1000
    knn_key_dim = 100
    knn_dict_capacity = 100 * scale
    knn_dict_update_step = 0.1 * scale
    knn_dict_delta = 0.001
    knn_dict_alpha = 0.1

start = time.time()

annoy_dict = Annoy_Dict(config)

for i in range(1200):
    start_ = time.time()
    embs = np.random.randn(int(config.knn_dict_update_step), config.knn_key_dim)
    values = np.random.randint(0, 10, int(config.knn_dict_update_step))
    # print("values: ", values)
    embs_next = np.random.randn(int(config.knn_dict_update_step), config.knn_key_dim)
    annoy_dict.add(keys=embs, values=values, keys_next=embs_next)
    end_ = time.time()
    # print("iter: %d, time: %f" % (i, end_ - start_))
    # capacity = annoy_dict.capacity_
    # print("capacity: ", capacity)

for i in range(10):
    start__ = time.time()
    test_data = np.random.randn(4, config.knn_key_dim)
    # print("np.ndim(test_data):", np.ndim(test_data))
    embs, values, embs_next = annoy_dict.query(keys=test_data, k=2)
    # print("values: ", values)
    end__ = time.time()
    # print("iter: %d, query time: %f" % (i, end__ - start__))

end = time.time()

print("total time: %f " % (end - start))
