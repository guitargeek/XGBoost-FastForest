/**

MIT License

Copyright (c) 2019 Jonas Rembser

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

#include "fastforest.h"

#include <fstream>
#include <string>
#include <sstream>
#include <stdexcept>

using namespace fastforest;

void fastforest::details::softmaxTransformInplace(TreeEnsembleResponseType* out, int nOut) {
    // do softmax transformation inplace
    TreeEnsembleResponseType norm = 0.;
    int i = 0;
    for (; i < nOut; ++i) {
        auto& x = out[i];
        x = std::exp(x);
        norm += x;
    }
    i = 0;
    for (; i < nOut; ++i) {
        auto& x = out[i];
        x /= norm;
    }
}

std::vector<TreeEnsembleResponseType> fastforest::FastForest::softmax(const FeatureType* array, int nClasses) const {
    auto out = std::vector<TreeEnsembleResponseType>(nClasses);
    softmax(array, out.data(), nClasses);
    return out;
}

void fastforest::FastForest::softmax(const FeatureType* array, TreeEnsembleResponseType* out, int nClasses) const {
    if (nClasses <= 2) {
        throw std::runtime_error(std::string{"Error in FastForest::softmax : nClasses is set to "} +
                                 std::to_string(nClasses) + ", but it should be at least equal 3 for the " +
                                 " multiclassification to make sense.");
    }

    evaluate(array, out, nClasses);
    fastforest::details::softmaxTransformInplace(out, nClasses);
}

void fastforest::FastForest::evaluate(const FeatureType* array, TreeEnsembleResponseType* out, int nOut) const {
    if (rootIndices_.size() % nOut != 0) {
        throw std::runtime_error(std::string{"Error in FastForest::softmax : Forest has "} +
                                 std::to_string(rootIndices_.size()) + " trees, " + "which is not compatible with " +
                                 std::to_string(nOut) + " classes!");
    }

    int iRootIndex = 0;
    for (int index : rootIndices_) {
        do {
            auto r = rightIndices_[index];
            auto l = leftIndices_[index];
            index = array[cutIndices_[index]] > cutValues_[index] ? r : l;
        } while (index > 0);
        out[iRootIndex % nOut] += responses_[-index];
        ++iRootIndex;
    }
}

FastForest fastforest::load_bin(std::string const& txtpath) {
    FastForest ff;

    std::ifstream is(txtpath, std::ios::binary);

    int nRootNodes = ff.rootIndices_.size();
    int nNodes = ff.cutValues_.size();
    int nLeaves = ff.responses_.size();

    is.read((char*)&nRootNodes, sizeof(int));
    is.read((char*)&nNodes, sizeof(int));
    is.read((char*)&nLeaves, sizeof(int));

    ff.rootIndices_.resize(nRootNodes);
    ff.cutIndices_.resize(nNodes);
    ff.cutValues_.resize(nNodes);
    ff.leftIndices_.resize(nNodes);
    ff.rightIndices_.resize(nNodes);
    ff.responses_.resize(nLeaves);

    is.read((char*)ff.rootIndices_.data(), nRootNodes * sizeof(int));
    is.read((char*)ff.cutIndices_.data(), nNodes * sizeof(CutIndexType));
    is.read((char*)ff.cutValues_.data(), nNodes * sizeof(FeatureType));
    is.read((char*)ff.leftIndices_.data(), nNodes * sizeof(int));
    is.read((char*)ff.rightIndices_.data(), nNodes * sizeof(int));
    is.read((char*)ff.responses_.data(), nLeaves * sizeof(TreeResponseType));

    return ff;
}

void fastforest::FastForest::write_bin(std::string const& filename) const {
    std::ofstream os(filename, std::ios::binary);

    int nRootNodes = rootIndices_.size();
    int nNodes = cutValues_.size();
    int nLeaves = responses_.size();

    os.write((const char*)&nRootNodes, sizeof(int));
    os.write((const char*)&nNodes, sizeof(int));
    os.write((const char*)&nLeaves, sizeof(int));

    os.write((const char*)rootIndices_.data(), nRootNodes * sizeof(int));
    os.write((const char*)cutIndices_.data(), nNodes * sizeof(CutIndexType));
    os.write((const char*)cutValues_.data(), nNodes * sizeof(FeatureType));
    os.write((const char*)leftIndices_.data(), nNodes * sizeof(int));
    os.write((const char*)rightIndices_.data(), nNodes * sizeof(int));
    os.write((const char*)responses_.data(), nLeaves * sizeof(TreeResponseType));
    os.close();
}
