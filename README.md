# XGBoost-FastForest

Minimal library code to deploy [XGBoost](https://xgboost.readthedocs.io/en/latest/) models in C++.

[![Build Status](https://travis-ci.com/guitargeek/XGBoost-FastForest.svg?branch=master)](https://travis-ci.com/guitargeek/XGBoost-FastForest) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3819838.svg)](https://doi.org/10.5281/zenodo.3819838)

In science, it is very common to protoype algorithms with Python and then put them in production with fast C++ code.
Transitioning models from Python to C++ should be as easy as possible to make sure new ideas can be tried out rapidly.
The __FastForest__ library helps you to get your xgboost model into a C++ production environment as quickly as possible.

The mission of this library is to be:
* __Easy__: deploying your xgboost model should be as painless as it can be
* __Fast__: thanks to efficient data structures for storing the trees, this library goes easy on your CPU and memory
* __Safe__: the FastForest objects are not mutated when used, and therefore they are an excellent choice in multithreading
  environments
* __Portable__: FastForest has no dependency other than the C++ standard library

### Installation

You can clone this repository, compile and install the library with __cmake__:
```
git clone git@github.com:guitargeek/FastForest.git
mkdir build
cd build
cmake ..
make
sudo make install
```

### Usage Example

Usually, xgboost models are trained via the __scikit-learn__ interface, like in this example with a random toy dataset.
At the end, we save the model both in __binary format__ to be able to still read it with xgboost, as well as in __text
format__ so we can open it with FastForest.

```Python
from xgboost import XGBClassifier
from sklearn.datasets import make_classification
import numpy as np

X, y = make_classification(n_samples=10000, n_features=5, random_state=42, n_classes=2, weights=[0.5])

model = XGBClassifier().fit(X, y)
booster = model._Booster

booster.dump_model("model.txt")
booster.save_model("model.bin")
```

In C++, you can now easily load the model into a `FastForest` and obtain predictions by calling the FastForest object with an array of features.

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

* You need to pass the names of the features that you will later use for the prediction to the FastForest constructor. This is necessary because the features are not ordered in the text file, hence you need to define an
  order yourself.
* Alternatively, can let the FastForest automatically determine an order by just passing an empty vector of strings. You will see the vector is filled with automatically determined feature names afterwards.
  * The original order of the features used in the training can't be recovered.
* The FastForest does not apply the [logistic transformation](https://en.wikipedia.org/wiki/Logistic_function).
  This is intentional, so you will not have any precision loss when you need the untransformed output. Thereforey ou need to apply
  the logistic transformation manually if you trained with `objective='binary:logistic'` and want to reproduce the results of `predict_proba()`, like in the code snippet above.
  * If you train with the `objective='binary:logitraw'`
    parameter, the output you'll get from `predict_proba()` will be without the logistic transformation, just like from the FastForest.

### Multiclassification with softmax

It is easily possible to use multiclassification models trained with the `multi:softmax` objective.

In this case, you should use the `FastForest::softmax` function. In addition to the features, you need to pass
the number of classes explicitely because this information is also not stored in the text dump of the model.

The function will return you a vector with the probabilites, one entry for each class.

```C++
std::vector<float> probas = fastForest.softmax(input.data(), 3);
```

For performance critical applications, this interface should not be used to avoid heap allocations in the vector
construction. Please use either the interface that requires to know the number of classes at compile-time as a template
parameter, or the old-school interface that writes the output into a function parameter.

```C++
{
  std::array<float,3> probas = fastForest.softmax<3>(input.data());
}
// or
{
  std::vector<float> probas(3); // allocated somewhere outside your loop over entries
  fastForest.softmax(input.data(), probas.data(), 3);
}
```

### Performance Benchmarks

So far, FastForest has been bencharked against the inference engine in the xgboost python library (undelying
C) and the [TMVA framework](https://root.cern.ch/tmva). For every engine, the same tree ensemble of 1000 trees is used,
and inference is done on a **single thread**.

| Engine                                                                                                  | Benchmark time   |
| :------                                                                                                 | ---------------: |
| __FastForest__ (GCC 9.3.0)                                                                              | 0.63 s           |
| [__treelite__](https://github.com/dmlc/treelite) (GCC 9.3.0)                                            | 1.2 s            |
| [__m2cgen__](https://github.com/BayesWitnesses/m2cgen)                                                  | 1.6 s            |
| [__xgboost__](https://xgboost.readthedocs.io/en/latest/python/python_api.html) 0.90 in __Python__ 3.8.2 | 2.6 s            |
| ROOT 6.20/00 [__TMVA__](https://root.cern.ch/tmva)                                                      | 4.3 s            |

The benchmak can be reproduced with the files found in the [benchmark directory](benchmark). The python scripts have to be
run first as they also train and save the models. Input type from the code generated by __m2cgen__ was changed from
`double` to `float` for a better comparison with __FastForest__.

The tests were performed on a Intel(R) Core(TM) i7-7820HQ CPU @ 2.90GHz.

### Serialization

The FastForests can serialized to it's own binary format. The binary format exactly reflects the memory layout of the
FastForest class, so saving and loading is as fast as it can be. The serialization to file is done with the `write_bin`
method.
```C++
fastForest.write_bin("forest.bin");
```
The serialized FastForest can be read back with it's constructor, this time the one that does not take a reference to a
vector for the feature names.
```C++
const auto fastForest = fastforest::load_bin("forest.bin");
```
