import numpy as np
import time

from knn.q_annoy_dict import Q_Annoy_Dict
class config:
    scale = 1000
    knn_key_dim = 3000
    knn_dict_capacity = 10 * scale
    knn_dict_update_step = 0.1 * scale
    knn_dict_delta = 0.001
    knn_dict_alpha = 0.1
    display_step = 0.1 * scale

num_action = 4
q_annoy_dict = Q_Annoy_Dict(config, num_action=num_action)
start = time.time()
start_ = time.time()

for i in range(config.knn_dict_capacity):
    emb = np.random.randn(32, config.knn_key_dim).astype(np.float32)
    val = np.random.randint(0, 10, size=32)
    action = np.random.randint(0, num_action, size=32)
    termial = np.random.randint(0, 2, size=32)
    emb_next = np.random.randn(32, config.knn_key_dim).astype(np.float32)
    q_annoy_dict.add(emb, action, val, termial, emb_next)
    end_ = time.time()
    if (i + 1) % config.display_step == 0:
        print("step: %d, time: %f " % (i + 1, end_ - start_))
        print("action capacity: ", q_annoy_dict.action_capacity)
        start_ = time.time()


for j in range(2):
    embs = np.random.randn(32, config.knn_key_dim).astype(np.float32)

    embs, actions, vals, teriminal, embs_next = q_annoy_dict.query_(embs, 2)
    print("shape(embs): ", np.shape(embs))
    print("shape(actions): ", np.shape(actions))
    print(actions)
    print("shape(vals): ", np.shape(vals))
    print("shape(teriminal): ", np.shape(teriminal))
    print("shape(embs_next): ", np.shape(embs_next))

end = time.time()

print("total time %f" % (end - start))