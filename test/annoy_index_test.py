from annoy import AnnoyIndex
import numpy as np
import time
start = time.time()
data = np.random.randn(1000, 100).astype(np.float32)

index = AnnoyIndex(100, metric="euclidean")

for ind, data_ in enumerate(data):
    index.add_item(ind, data_)
index.build(50)
print(index.get_n_items())

data_1 = np.random.randn(1000, 100).astype(np.float32)
for ind, data_ in enumerate(data_1):
    index.add_item(ind + 1000, data_)

print(index.get_n_items())

index.build(50)
print(index.get_n_items())
end = time.time()

for ind, data_ in enumerate(data_1):
    index.add_item(ind + 1000, data_)
print(index.get_n_items())

index.build(50)
print(index.get_n_items())
print(end - start)

