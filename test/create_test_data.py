from xgboost import XGBClassifier
from sklearn.datasets import make_classification

import numpy as np
import pandas as pd

import os

csv_args = dict(header=False, index=False, sep=" ")


def create_test_data(X, y, directory, n_dump_samples=100):
    if not os.path.exists(directory):
        os.makedirs(directory)

    model = XGBClassifier(n_estimators=100, max_depth=7, objective="binary:logitraw").fit(X, y)

    model._Booster.dump_model(os.path.join(directory, "model.txt"))
    model._Booster.save_model(os.path.join(directory, "model.bin"))

    X_dump = X[:n_dump_samples]

    preds_dump = model.predict_proba(X_dump)[:, 1]

    pd.DataFrame(X_dump).to_csv(os.path.join(directory, "X.csv"), **csv_args)
    pd.DataFrame(preds_dump).to_csv(os.path.join(directory, "preds.csv"), **csv_args)


X, y = make_classification(n_samples=10000, n_features=5, random_state=42, n_classes=2, weights=[0.5])

create_test_data(X, y, "continuous")

X_discrete = np.array(X, dtype=np.int) * 2

create_test_data(X_discrete, y, "discrete")
# Why do we add +1 here? It's to cover the test case where integer features lie exactly on the cut values.
# For this trainings test, the features will all be even numbers to the cut values will be odd numbers.
# Therefore, we can add +1 to the features to get odd numbers as well and cover the test case of a feature
# being equal to a cut.
X_dump = pd.read_csv("discrete/X.csv", header=None, sep=" ") + 1
X_dump.to_csv("discrete/X.csv", **csv_args)

df = pd.read_csv("manyfeatures.csv.gz", header=None, compression="gzip")

_, y = make_classification(n_samples=len(df), random_state=43, n_classes=2, weights=[0.5])

create_test_data(df.values, y, "manyfeatures")
