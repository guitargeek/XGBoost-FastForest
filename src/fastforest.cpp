/**

MIT License

Copyright (c) 2025 Jonas Rembser

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#include <fastforest.h>

#include <cmath>
#include <fstream>
#include <string>
#include <sstream>
#include <stdexcept>

using namespace fastforest;

void fastforest::details::softmaxTransformInplace(TreeEnsembleResponseType* out, int nOut) {
    // Do softmax transformation inplace, mimicking exactly the Softmax function
    // in the src/common/math.h source file of xgboost.
    double norm = 0.;
    TreeEnsembleResponseType wmax = *out;
    int i = 1;
    for (; i < nOut; ++i) {
        wmax = std::max(out[i], wmax);
    }
    i = 0;
    for (; i < nOut; ++i) {
        TreeEnsembleResponseType& x = out[i];
        x = std::exp(x - wmax);
        norm += x;
    }
    i = 0;
    for (; i < nOut; ++i) {
        out[i] /= static_cast<float>(norm);
    }
}

std::vector<TreeEnsembleResponseType> fastforest::FastForest::softmax(const FeatureType* array,
                                                                      TreeEnsembleResponseType baseResponse) const {
    std::vector<TreeEnsembleResponseType> out(nClasses());
    softmax(array, out.data(), baseResponse);
    return out;
}

void fastforest::FastForest::softmax(const FeatureType* array,
                                     TreeEnsembleResponseType* out,
                                     TreeEnsembleResponseType baseResponse) const {
    int nClass = nClasses();
    if (nClass <= 2) {
        throw std::runtime_error(
            "Error in FastForest::softmax : binary classification models don't support softmax evaluation. Please set "
            "the number of classes in the FastForest-creating function if this is a multiclassification model.");
    }

    evaluate(array, out, nClass, baseResponse);
    fastforest::details::softmaxTransformInplace(out, nClass);
}

void fastforest::FastForest::evaluate(const FeatureType* array,
                                      TreeEnsembleResponseType* out,
                                      int nOut,
                                      TreeEnsembleResponseType baseResponse) const {
    for (int i = 0; i < nOut; ++i) {
        out[i] = baseResponse + baseResponses_[i];
    }

    int iRootIndex = 0;
    for (std::vector<int>::const_iterator indexIter = rootIndices_.begin(); indexIter != rootIndices_.end();
         ++indexIter) {
        int index = *indexIter;
        do {
            int r = rightIndices_[index];
            int l = leftIndices_[index];
            index = array[cutIndices_[index]] < cutValues_[index] ? l : r;
        } while (index > 0);
        out[treeNumbers_[iRootIndex] % nOut] += responses_[-index];
        ++iRootIndex;
    }
}

TreeEnsembleResponseType fastforest::FastForest::evaluateBinary(const FeatureType* array,
                                                                TreeEnsembleResponseType baseResponse) const {
    TreeEnsembleResponseType out = baseResponse + baseResponses_[0];

    for (std::vector<int>::const_iterator indexIter = rootIndices_.begin(); indexIter != rootIndices_.end();
         ++indexIter) {
        int index = *indexIter;
        do {
            int r = rightIndices_[index];
            int l = leftIndices_[index];
            index = array[cutIndices_[index]] < cutValues_[index] ? l : r;
        } while (index > 0);
        out += responses_[-index];
    }

    return out;
}

FastForest fastforest::load_bin(std::string const& txtpath) {
    std::ifstream ifs(txtpath.c_str(), std::ios::binary);
    return load_bin(ifs);
}

FastForest fastforest::load_bin(std::istream& is) {
    FastForest ff;

    int nRootNodes;
    int nNodes;
    int nLeaves;

    is.read((char*)&nRootNodes, sizeof(int));
    is.read((char*)&nNodes, sizeof(int));
    is.read((char*)&nLeaves, sizeof(int));

    ff.rootIndices_.resize(nRootNodes);
    ff.cutIndices_.resize(nNodes);
    ff.cutValues_.resize(nNodes);
    ff.leftIndices_.resize(nNodes);
    ff.rightIndices_.resize(nNodes);
    ff.responses_.resize(nLeaves);
    ff.treeNumbers_.resize(nRootNodes);

    is.read((char*)ff.rootIndices_.data(), nRootNodes * sizeof(int));
    is.read((char*)ff.cutIndices_.data(), nNodes * sizeof(CutIndexType));
    is.read((char*)ff.cutValues_.data(), nNodes * sizeof(FeatureType));
    is.read((char*)ff.leftIndices_.data(), nNodes * sizeof(int));
    is.read((char*)ff.rightIndices_.data(), nNodes * sizeof(int));
    is.read((char*)ff.responses_.data(), nLeaves * sizeof(TreeResponseType));
    is.read((char*)ff.treeNumbers_.data(), nRootNodes * sizeof(int));

    int nBaseResponses;
    is.read((char*)&nBaseResponses, sizeof(int));
    ff.baseResponses_.resize(nBaseResponses);
    is.read((char*)ff.baseResponses_.data(), nBaseResponses * sizeof(TreeEnsembleResponseType));

    return ff;
}

void fastforest::FastForest::write_bin(std::string const& filename) const {
    std::ofstream os(filename.c_str(), std::ios::binary);

    int nRootNodes = rootIndices_.size();
    int nNodes = cutValues_.size();
    int nLeaves = responses_.size();
    int nBaseResponses = baseResponses_.size();

    os.write((const char*)&nRootNodes, sizeof(int));
    os.write((const char*)&nNodes, sizeof(int));
    os.write((const char*)&nLeaves, sizeof(int));

    os.write((const char*)rootIndices_.data(), nRootNodes * sizeof(int));
    os.write((const char*)cutIndices_.data(), nNodes * sizeof(CutIndexType));
    os.write((const char*)cutValues_.data(), nNodes * sizeof(FeatureType));
    os.write((const char*)leftIndices_.data(), nNodes * sizeof(int));
    os.write((const char*)rightIndices_.data(), nNodes * sizeof(int));
    os.write((const char*)responses_.data(), nLeaves * sizeof(TreeResponseType));
    os.write((const char*)treeNumbers_.data(), nRootNodes * sizeof(int));

    os.write((const char*)&nBaseResponses, sizeof(int));
    os.write((const char*)baseResponses_.data(), nBaseResponses * sizeof(TreeEnsembleResponseType));
    os.close();
}
