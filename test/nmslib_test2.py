import nmslib
import numpy as np
data = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
data = np.array(data)

index = nmslib.init(method="hnsw", space="cosinesimil")
index.addDataPointBatch(data=data, ids=[0, 1, 2])
index.createIndex()

result_1 = index.knnQuery(data[0], k=1)
print("result_1 ", result_1)
data_1 = [[13, 14, 15, 16],
          [17, 18, 19, 20],
          [21, 22, 23, 24]]
data_1 = np.array(data_1)


index.addDataPointBatch(data=data_1, ids=[3, 4, 5])
result_2 = index.knnQuery(data[0], k=1)
print("result_2 ", result_2)

result_3 = index.knnQuery(data_1[0], k=1)
print("result_3 ", result_3)
index.createIndex()
result_4 = index.knnQuery(data_1[0], k=1)
print("result_4 ", result_4)
data_3 =[25, 26, 27, 28]
index.addDataPoint(data=data_3, id=2)
result_5 = index.knnQuery(data[0], k=1)
print("result_5 ", result_5)
index.createIndex()
result_6 = index.knnQuery(data[0], k=1)
print("result_6 ", result_6)

index.addDataPointBatch(data=data_1, ids=[0, 1, 2])
index.createIndex()
result_7 = index.knnQuery(data[0], k=1)
print("result_7 ", result_7)

print(type(index))