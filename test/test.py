import numpy as np

data = [[[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
        [[9, 10], [11, 12]],
        [[13, 14], [15, 16]]]
print(np.shape(data))

print(data[:-1])
print(data[1:])
data[:-1] = data[1:]
data[-1] = [[1, 2], [3, 4]]
print(data)