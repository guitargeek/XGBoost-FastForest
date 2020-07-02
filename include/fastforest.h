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

#ifndef FastForest_h
#define FastForest_h

#include <vector>
#include <string>
#include <array>
#include <cmath>

namespace fastforest {

    // The floating point number type that will be used to accept features and store cut values
    typedef float FeatureType;
    // Tue floating point number type that the individual trees return their responses in
    typedef float TreeResponseType;
    // The floating point number type that is used to sum the individual tree responses
    typedef float TreeEnsembleResponseType;
    // This integer type stores the indices of the feature employed in each cut.
    // Set to `unsigned char` for most compact fastforest ofjects if you have less than 256 features.
    typedef unsigned int CutIndexType;

    namespace details {

        void softmaxTransformInplace(TreeEnsembleResponseType* out, int nOut);

    }

    struct FastForest {
        TreeEnsembleResponseType operator()(const FeatureType* array) const {
            TreeEnsembleResponseType out{0.};
            evaluate(array, &out);
            return out;
        }

        template <int nClasses>
        std::array<TreeEnsembleResponseType, nClasses> softmax(const FeatureType* array) const {
            // static softmax interface: no manual memory allocation, but requires to know nClasses at compile time
            static_assert(nClasses >= 3, "nClasses should be >= 3");
            std::array<TreeEnsembleResponseType, nClasses> out{};
            evaluate(array, out.data(), nClasses);
            details::softmaxTransformInplace(out.data(), nClasses);
            return out;
        }
        // dynamic softmax interface with manually allocated std::vector: simple but inefficient
        std::vector<TreeEnsembleResponseType> softmax(const FeatureType* array, int nClasses) const;
        // softmax interface that is not a pure function, but no manual allocation and no compile-time knowledge needed
        void softmax(const FeatureType* array, TreeEnsembleResponseType* out, int nClasses) const;

        void write_bin(std::string const& filename) const;

        std::vector<int> rootIndices_;
        std::vector<CutIndexType> cutIndices_;
        std::vector<FeatureType> cutValues_;
        std::vector<int> leftIndices_;
        std::vector<int> rightIndices_;
        std::vector<TreeResponseType> responses_;

      private:
        void evaluate(const FeatureType* array, TreeEnsembleResponseType* out, int nOut = 1) const;
    };

    FastForest load_txt(std::string const& txtpath, std::vector<std::string>& features);
    FastForest load_bin(std::string const& txtpath);
#ifdef EXPERIMENTAL_TMVA_SUPPORT
    FastForest load_tmva_xml(std::string const& xmlpath, std::vector<std::string>& features);
#endif

}  // namespace fastforest

#endif
