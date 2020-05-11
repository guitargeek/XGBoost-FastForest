from xgboost import XGBClassifier
from sklearn.datasets import make_classification
import numpy as np
import time
import sys

import xgboost2tmva
import m2cgen as m2c

sys.setrecursionlimit(1000000)

X, y = make_classification(n_samples=10000, n_features=5, random_state=42, n_classes=2, weights=[0.5])

model = XGBClassifier(n_estimators=1000, objective="binary:logistic").fit(X, y)

model._Booster.dump_model("model.txt")
model._Booster.save_model("model.bin")

# export to TMVA-style XML file
input_variables = [("f" + str(i), "F") for i in range(5)]
xgboost2tmva.convert_model(model._Booster.get_dump(), input_variables, "model.xml")

# export to hardcoded C
code = m2c.export_to_c(model)
with open("model.c", "w") as c_file:
    c_file.write(code)

X_test = np.random.uniform(-5, 5, size=(100000, 5))

start_time = time.time()

preds = model.predict_proba(X_test)[:, 1]
print(np.mean(preds))

elapsed_secs = time.time() - start_time

print("Wall time for inference: {0:.2f} s".format(elapsed_secs))
