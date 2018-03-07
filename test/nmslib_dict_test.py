from knn.nmslib_dict import NMSlib_Dict
import numpy as np
import time
class config:
    scale = 100
    id_knn_dict_used = False
    knn_key_dim = 100
    knn_dict_capacity = 10 * scale
    knn_dict_update_step = 1 * scale
    knn_dict_delta = 0.001
    knn_dict_alpha = 0.1

    knn_dict_memory_replay_sample_concate = False
    knn_dict_memory_replay_sample_concate_rate = 0.
    nmslib_print_progress = True

start = time.time()

nmslib_dict = NMSlib_Dict(config)

for i in range(11):
    start_ = time.time()
    embs = np.random.rand(int(config.knn_dict_update_step), config.knn_key_dim)
    values = np.random.randint(0, 10, int(config.knn_dict_update_step))
    print("values: ", values)
    embs_next = np.random.rand(int(config.knn_dict_update_step), config.knn_key_dim)
    nmslib_dict.add(embs=embs, values=values, embs_next=embs_next)
    end_ = time.time()
    print("iter: %d, time: %f" % (i, end_ - start_))

for i in range(10):
    start__ = time.time()
    test_data = np.random.randn(4, config.knn_key_dim)
    print("np.ndim(test_data):", np.ndim(test_data))
    embs, values, embs_next = nmslib_dict.query_batch(test_data, k=1)
    print(values)
    end__ = time.time()
    print("iter: %d, query time: %f" % (i, end__ - start__))

end = time.time()

print("total time: %f " % (end - start))
