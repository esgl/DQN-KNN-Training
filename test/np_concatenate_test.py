import numpy as np
a = np.random.randn(2, 3)
b = np.random.randn(2, 3)
c = np.random.randn(2, 3)
print(a)
print(b)
print(c)
d = np.concatenate((a, b, c), axis=0)
print(d)