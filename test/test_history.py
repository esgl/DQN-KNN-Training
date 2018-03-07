import numpy as np
a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
a = np.array(a)
count = a.shape[0]
history_length = 4

def getData(index):
    if index >= history_length - 1:
        return a[(index - (history_length - 1)) : (index + 1)]
    else:
        indexes = [(index - i) % count for i in reversed(range(history_length))]
        return a[indexes]

for index in range(count):
    print(index)
    print(getData(index))
