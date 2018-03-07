import numpy as np
a = [[1, 2, 3, 4], [5, 6, 7, 8]]

a_n = np.array(a)

for idx, a_ in enumerate(a):
    print(type(a_))
    print(idx, a_)

print("###########################")
for idx, a_ in enumerate(a_n):
    print(type(a_))
    print(idx, a_)