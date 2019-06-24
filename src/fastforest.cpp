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

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <unordered_map>
#include <stdexcept>

namespace {

    namespace util {

        inline bool isInteger(const std::string& s) {
            if (s.empty() || ((!isdigit(s[0])) && (s[0] != '-') && (s[0] != '+')))
                return false;

            char* p;
            strtol(s.c_str(), &p, 10);

            return (*p == 0);
        }

        template <class NumericType>
        struct NumericAfterSubstrOutput {
            NumericType value = 0;
            bool found = false;
            bool failed = true;
            std::string rest;
        };

        template <class NumericType>
        inline auto numericAfterSubstr(std::string const& str, std::string const& substr) {
            NumericType x = 0;
            std::string rest;
            NumericAfterSubstrOutput<NumericType> output;
            output.rest = str;

            auto found = str.find(substr);
            if (found != std::string::npos) {
                output.found = true;
                std::stringstream ss(str.substr(found + substr.size(), str.size() - found + substr.size()));
                ss >> output.value;
                if (!ss.fail()) {
                    output.failed = false;
                    output.rest = ss.str();
                }
            }
            return output;
        }

        std::vector<std::string> split(std::string const& strToSplit, char delimeter) {
            std::stringstream ss(strToSplit);
            std::string item;
            std::vector<std::string> splittedStrings;
            while (std::getline(ss, item, delimeter)) {
                splittedStrings.push_back(item);
            }
            return splittedStrings;
        }
    }  // namespace util

    namespace detail {
        void correctIndices(std::vector<int>& txtIndices,
                            std::unordered_map<int, int> const& nodeIndices,
                            std::unordered_map<int, int> const& leafIndices) {
            for (auto& i : txtIndices) {
                if (nodeIndices.count(i)) {
                    i = nodeIndices.at(i);
                } else if (leafIndices.count(i)) {
                    i = -leafIndices.at(i);
                } else {
                    throw std::runtime_error("something is wrong in the node structure");
                }
            }
        }
    }  // namespace detail

}  // namespace

FastForest::FastForest(std::string const& txtpath, std::vector<std::string>& features) {
    const std::string info = "constructing FastForest from " + txtpath + ": ";

    FastTree* tree = nullptr;

    std::ifstream file(txtpath);

    int nVariables = 0;
    std::unordered_map<std::string, int> varIndices;
    bool fixFeatures = false;

    if (!features.empty()) {
        fixFeatures = true;
        nVariables = features.size();
        for (int i = 0; i < nVariables; ++i) {
            varIndices[features[i]] = i;
        }
    }

    std::string line;

    std::unordered_map<int, int> nodeIndices;
    std::unordered_map<int, int> leafIndices;

    while (std::getline(file, line)) {
        auto foundBegin = line.find("[");
        auto foundEnd = line.find("]");
        if (foundBegin != std::string::npos) {
            auto subline = line.substr(foundBegin + 1, foundEnd - foundBegin - 1);
            if (util::isInteger(subline)) {
                if (tree) {
                    detail::correctIndices(tree->rightIndices_, nodeIndices, leafIndices);
                    detail::correctIndices(tree->leftIndices_, nodeIndices, leafIndices);
                }
                trees_.emplace_back();
                tree = &trees_.back();
                nodeIndices.clear();
                leafIndices.clear();
            } else {
                std::stringstream ss(line);
                int index;
                ss >> index;
                line = ss.str();

                auto splitstring = util::split(subline, '<');
                auto const& varName = splitstring[0];
                float cutValue = stof(splitstring[1]);
                if (!varIndices.count(varName)) {
                    if (fixFeatures) {
                        throw std::runtime_error(info + "feature " + varName + " not in list of features");
                    }
                    varIndices[varName] = nVariables;
                    features.push_back(varName);
                    ++nVariables;
                }
                int yes;
                int no;
                auto output = util::numericAfterSubstr<int>(line, "yes=");
                if (!output.failed) {
                    yes = output.value;
                } else {
                    throw std::runtime_error(info + "problem while parsing the text dump");
                }
                output = util::numericAfterSubstr<int>(output.rest, "no=");
                if (!output.failed) {
                    no = output.value;
                } else {
                    throw std::runtime_error(info + "problem while parsing the text dump");
                }

                tree->cutValues_.push_back(cutValue);
                tree->cutIndices_.push_back(varIndices[varName]);
                tree->leftIndices_.push_back(yes);
                tree->rightIndices_.push_back(no);
                nodeIndices[index] = nodeIndices.size();
            }

        } else {
            auto output = util::numericAfterSubstr<float>(line, "leaf=");
            if (output.found) {
                std::stringstream ss(line);
                int index;
                ss >> index;
                line = ss.str();

                tree->responses_.push_back(output.value);
                leafIndices[index] = leafIndices.size();
            }
        }
    }
    if (tree) {
        detail::correctIndices(tree->rightIndices_, nodeIndices, leafIndices);
        detail::correctIndices(tree->leftIndices_, nodeIndices, leafIndices);
    }
}
