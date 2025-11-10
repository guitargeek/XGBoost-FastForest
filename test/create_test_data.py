import xgboost as xgb
from sklearn.datasets import make_classification

import numpy as np
import pandas as pd

import os

csv_args = dict(header=False, index=False, sep=" ")


def get_basescore(model):
    """Get base score from an XGBoost sklearn estimator.

    Copy-pasted from XGBoost unit test code.

    See also:
      * https://github.com/dmlc/xgboost/blob/2463938/python-package/xgboost/testing/updater.py#L43
      * https://github.com/dmlc/xgboost/issues/9347
      * https://discuss.xgboost.ai/t/how-to-get-base-score-from-trained-booster/3192
    """
    import json

    jintercept = json.loads(model.get_booster().save_config())["learner"]["learner_model_param"]["base_score"]
    out = json.loads(jintercept)
    if isinstance(out, float):
        # Before XGBoost 3.1.0, this was a single float. So we pack it into a
        # list ourselves.
        return [out]
    return out


def create_test_data(
    X, y, directory, n_dump_samples=100, objective="binary:logitraw", eval_metric="logloss", convert_to_tmva=False
):
    if not os.path.exists(directory):
        os.makedirs(directory)

    model = xgb.XGBClassifier(n_estimators=100, max_depth=7, objective=objective, eval_metric=eval_metric).fit(X, y)

    outfile_json = os.path.join(directory, "model.json")

    # Make sure JSON roundtripping works (broken in XGBoost 2.0.3)
    if xgb.__version__ != "2.0.3":
        model.save_model(outfile_json)

        model = xgb.XGBClassifier()
        model.load_model(outfile_json)

    outfile = os.path.join(directory, "model.txt")
    booster = model.get_booster()
    # Dump the model to a .txt file
    booster.dump_model(outfile, fmap="", with_stats=False, dump_format="text")
    # Append the base score (unfortunately missing in the .txt dump)
    with open(outfile, "a") as f:
        base_score = get_basescore(model)
        f.write(f"base_score={base_score}\n")

    if int(xgb.__version__[0]) < 2:
        # Replace all '<' with '<=' in the text dump file (before version 2.0,
        # XGBoost used inconsistent comparison operators in the model).

        with open(outfile, "r") as f:
            text = f.read()

        with open(outfile, "w") as f:
            f.write(text.replace("<", "<="))

    if convert_to_tmva:
        import xgboost2tmva

        feature_names = [(f"f{i}", "F") for i in range(len(y))]
        xgboost2tmva.convert_model(model._Booster.get_dump(), feature_names, os.path.join(directory, "model.xml"))

    X_dump = X[:n_dump_samples]

    preds_dump = model.predict_proba(X_dump)

    if preds_dump.shape[1] == 2:
        preds_dump = preds_dump[:, 1]

    pd.DataFrame(X_dump).to_csv(os.path.join(directory, "X.csv"), **csv_args)
    pd.DataFrame(preds_dump).to_csv(os.path.join(directory, "preds.csv"), **csv_args)


def main():

    n_features = 5
    X, y = make_classification(n_samples=10000, n_features=n_features, random_state=42, n_classes=2, weights=[0.5])

    create_test_data(X, y, "continuous")

    X_discrete = np.array(X, dtype=int) * 2

    create_test_data(X_discrete, y, "discrete")
    # Why do we add +1 here? It's to cover the test case where integer features
    # lie exactly on the cut values. For this trainings test, the features will
    # all be even numbers to the cut values will be odd numbers. Therefore, we
    # can add +1 to the features to get odd numbers as well and cover the test
    # case of a feature being equal to a cut.
    X_dump = pd.read_csv("discrete/X.csv", header=None, sep=" ") + 1
    X_dump.to_csv("discrete/X.csv", **csv_args)

    df = pd.read_csv("manyfeatures.csv.gz", header=None, compression="gzip")

    _, y = make_classification(n_samples=len(df), random_state=43, n_classes=2, weights=[0.5])

    create_test_data(df.values, y, "manyfeatures")

    # for multiclassification
    X, y = make_classification(
        n_samples=10000, n_features=5, n_informative=3, random_state=42, n_classes=3, weights=[0.33, 0.33]
    )
    create_test_data(X, y, "softmax", objective="multi:softproba", eval_metric="mlogloss")

    # for GitHub issue #15
    X, y = make_classification(
        n_samples=100, n_features=100, n_informative=3, random_state=42, n_classes=3, weights=[0.33, 0.33]
    )
    create_test_data(X, y, "softmax_n_samples_100_n_features_100", objective="multi:softproba", eval_metric="mlogloss")


if __name__ == "__main__":
    main()
