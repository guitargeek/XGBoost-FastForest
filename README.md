# XGBoost-FastForest

Minimal library code to deploy [XGBoost](https://xgboost.readthedocs.io/en/latest/) models in C++.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8400945.svg)](https://doi.org/10.5281/zenodo.8400945)

In science, it is very common to prototype algorithms with Python and then put them in production with fast C++ code.
Transitioning models from Python to C++ should be as easy as possible to make sure new ideas can be tried out rapidly.
The __FastForest__ library helps you to get your XGBoost model into a C++ production environment as quickly as possible.

The mission of this library is to be:
* __Easy__: deploying your XGBoost model should be as painless as it can be
* __Fast__: thanks to efficient data structures for storing the trees, this library goes easy on your CPU and memory
* __Safe__: the FastForest objects are not mutated when used, and therefore they are an excellent choice in multithreading
  environments
* __Portable__: FastForest has no dependency other than the C++ standard library, and the minimum required C++ standard is C++98

### Installation

You can clone this repository, compile and install the library with __cmake__:
```Bash
git clone git@github.com:guitargeek/FastForest.git
mkdir build
cd build
cmake ..
make
sudo make install
```

### Usage Example

Usually, XGBoost models are trained via the __scikit-learn__ interface, like in this example with a random toy dataset.
In the end, we save the model both in __binary format__ to be able to still read it with XGBoost, as well as in __text
format__ so we can open it with FastForest.

```Python
import xgboost
from xgboost import XGBClassifier
from sklearn.datasets import make_classification
import numpy as np

X, y = make_classification(
    n_samples=10000, n_features=5, random_state=42, n_classes=2, weights=[0.5]
)

model = XGBClassifier().fit(X, y)

outfile = "model.txt"
booster = model.get_booster()
# Dump the model to a .txt file
booster.dump_model(outfile, fmap=, with_stats=False, dump_format="text")
# Append the base score (unfortunately missing in the .txt dump)
with open(outfile, "a") as f:
    import json

    json_dump = json.loads(booster.save_config())
    base_score = json_dump["learner"]["learner_model_param"]["base_score"]
    f.write(f"base_score={base_score}\n")   f.write(f"xgboost_version={xgboost.__version__}\n")
```

In the future, FastForest might migrate to XGBoosts `.json` format for the model input, since this schema encodes the base score as well.

In C++, you can now quickly load the model into a `FastForest` and obtain predictions by calling the FastForest object with an array of features.

```C++
#include "fastforest.h"
#include <cmath>

int main() {
    std::vector<std::string> features{"f0",  "f1",  "f2",  "f3",  "f4"};

    const auto fastForest = fastforest::load_txt("model.txt", features);

    std::vector<float> input{0.0, 0.2, 0.4, 0.6, 0.8};

    float score = 1./(1. + std::exp(-fastForest(input.data())));
}
```

Some things to keep in mind:

* You need to pass the names of the features that you will later use for the prediction to the FastForest constructor. This argument is necessary because the features are not ordered in the text file. Hence you need to define an order yourself.
* Alternatively, can let the FastForest automatically determine an order by just passing an empty vector of strings. You will see the vector is filled with automatically determined feature names afterwards.
  * The original order of the features used in the training process can't be recovered.
* The FastForest does not apply the [logistic transformation](https://en.wikipedia.org/wiki/Logistic_function), so you will not have any precision loss when you need the untransformed output. Therefore, you need to apply
  the logistic transformation manually if you trained with `objective='binary:logistic'` and want to reproduce the results of `predict_proba()`, like in the code snippet above.
  * If you train with the `objective='binary:logitraw'`
    parameter, the output you'll get from `predict_proba()` will be without the logistic transformation, just like from the FastForest.

### Multiclass classification with softmax

It is easily possible to use multiclassification models trained with the `multi:softmax` objective.

In this case, you should pass the number of classes to `fastforest::load_txt` and use the `FastForest::softmax` function for evaluation.

The function will return you a vector with the probabilities, one entry for each class.

```C++
std::vector<float> probas = fastForest.softmax(input.data());
```

For performance-critical applications, this interface should not be used to avoid heap allocations in the vector
construction. Please use either the `std::array` interface or the old-school interface that writes the output into a function parameter.

```C++
{
  std::array<float,3> probas = fastForest.softmax<3>(input.data());
}
// or
{
  std::vector<float> probas(3); // allocated somewhere outside your loop over entries
  fastForest.softmax(input.data(), probas.data());
}
```

### Performance Benchmarks

So far, FastForest has been benchmarked against the inference engine in the XGBoost python library (underlying
C) and the [TMVA framework](https://root.cern.ch/tmva). For every engine, the same tree ensemble of 1000 trees is used,
and inference is made on a **single thread**.

| Engine                                                                                                    | Benchmark time   |
| :------                                                                                                   | ---------------: |
| __FastForest__ (GCC 11.1.0)                                                                               | 1.5 s            |
| [__treelite__](https://github.com/dmlc/treelite) (GCC 11.1.0)                                             | 2.7 s            |
| [__m2cgen__](https://github.com/BayesWitnesses/m2cgen) (GCC 11.1.0 with `-O1`[^1])                        | 1.3 s            |
| [__xgboost__](https://xgboost.readthedocs.io/en/latest/python/python_api.html) 1.5.2 in __Python__ 3.10.1 | 2.0 s            |
| ROOT 6.24/00 [__TMVA__](https://root.cern.ch/tmva)                                                        | 5.6 s            |

The benchmark can be reproduced with the files found in the [benchmark directory](benchmark). The python scripts have to be
run first as they also train and save the models. Input type from the code generated by __m2cgen__ was changed from
`double` to `float` for a better comparison with __FastForest__.

The tests were performed on an AMD Ryzen 9 3900 12-Core Processor.

[^1]: Different optimization flags were compared, and `-O1` was by far the best for the `m2cgen` model performance. Also note that the performance of the `m2cgen` approach of hardcoding the model as C code is very sensitive to the compiler. In particular, older compilers result in slower evaluation.

### Serialization

The FastForests can be serialized to binary files. The binary format reflects the memory layout of the FastForest class, so saving and loading is as fast as it can be. The serialization to file is done with the write_bin method.
```C++
fastForest.write_bin("forest.bin");
```
The serialized FastForest can be read back with its constructor, this time the one that does not take a reference to a vector for the feature names.

```C++
const auto fastForest = fastforest::load_bin("forest.bin");
```
