import numpy as np
import nmslib
from multiprocessing import cpu_count
np.random.seed(123)

data = np.random.randn(400, 84 * 84).astype(np.float16)
index = nmslib.init(method="hnsw", space="cosinesimil")
ids = np.arange(0, data.shape[0])
index.addDataPointBatch(data, ids)
index.createIndex(print_progress=True)
test = np.random.randn(4, 84*84).astype(np.float16)


results = index.knnQueryBatch(queries=test, k=3, num_threads=cpu_count())
results = np.array(results, dtype=np.int)
print(results[:,0,:])
print(results[:,0,:])
print(results[:,1,:])

test_t = test[0,...]
result_t = index.knnQuery(vector=test_t, k=3)
result_t_ind = np.array(result_t, dtype=np.int)[0]
result_t_dist = np.array(result_t, dtype=np.float64)[1]
print(result_t)
print(result_t_ind)
print(result_t_dist)
