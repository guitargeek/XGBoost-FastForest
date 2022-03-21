import matplotlib.pyplot as plt
import numpy as np
import cppyy
import pandas as pd
import ctypes

cppyy.include('fastforest.h')
cppyy.load_library('/usr/local/lib/libfastforest.so')

X = pd.read_csv("continuous/X.csv", header=None, delimiter=" ").to_numpy(dtype=np.float32)
preds_ref = pd.read_csv("continuous/preds.csv", header=None, delimiter=" ").to_numpy().T[0]

print(X)
print(preds_ref)

features = cppyy.gbl.std.vector["std::string"]()
for f in ("f0", "f1", "f2", "f3", "f4"):
    features.push_back(f)

fast_forest = cppyy.gbl.fastforest.load_txt("continuous/model.txt", features)

preds_ff = np.zeros_like(preds_ref, dtype=np.float32)

for i in range(X.shape[0]):
    tmp = X[i].copy()
    arr = tmp.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    preds_ff[i] = fast_forest(arr)

print(preds_ff)

plt.scatter(preds_ref, preds_ff)
plt.show()
