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
#include "common_details.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <stdexcept>
#include <iostream>
#include <stdlib.h> /* strtol */

using namespace fastforest;

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
            explicit NumericAfterSubstrOutput() {
                value = 0;
                found = false;
                failed = true;
            }
            NumericType value;
            bool found;
            bool failed;
            std::string rest;
        };

        template <class NumericType>
        inline NumericAfterSubstrOutput<NumericType> numericAfterSubstr(std::string const& str,
                                                                        std::string const& substr) {
            std::string rest;
            NumericAfterSubstrOutput<NumericType> output;
            output.rest = str;

            std::size_t found = str.find(substr);
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

        std::vector<std::string> split(std::string const& strToSplit, char delimiter) {
            std::stringstream ss(strToSplit);
            std::string item;
            std::vector<std::string> splitStrings;
            while (std::getline(ss, item, delimiter)) {
                splitStrings.push_back(item);
            }
            return splitStrings;
        }

        bool exists(std::string const& filename) {
            if (FILE* file = fopen(filename.c_str(), "r")) {
                fclose(file);
                return true;
            } else {
                return false;
            }
        }

    }  // namespace util

    void terminateTree(fastforest::FastForest& ff,
                       int& nPreviousNodes,
                       int& nPreviousLeaves,
                       fastforest::detail::IndexMap& nodeIndices,
                       fastforest::detail::IndexMap& leafIndices,
                       int& treesSkipped) {
        using namespace fastforest::detail;
        correctIndices(ff.rightIndices_.begin() + nPreviousNodes, ff.rightIndices_.end(), nodeIndices, leafIndices);
        correctIndices(ff.leftIndices_.begin() + nPreviousNodes, ff.leftIndices_.end(), nodeIndices, leafIndices);

        if (nPreviousNodes != ff.cutValues_.size()) {
            ff.treeNumbers_.push_back(ff.rootIndices_.size() + treesSkipped);
            ff.rootIndices_.push_back(nPreviousNodes);
        } else {
            int treeNumbers = ff.rootIndices_.size() + treesSkipped;
            ++treesSkipped;
            ff.baseResponses_[treeNumbers % ff.baseResponses_.size()] += ff.responses_.back();
            ff.responses_.pop_back();
        }

        nodeIndices.clear();
        leafIndices.clear();
        nPreviousNodes = ff.cutValues_.size();
        nPreviousLeaves = ff.responses_.size();
    }

}  // namespace

FastForest fastforest::load_txt(std::string const& txtpath, std::vector<std::string>& features, int nClasses) {
    const std::string info = "constructing FastForest from " + txtpath + ": ";

    if (!util::exists(txtpath)) {
        throw std::runtime_error(info + "file does not exists");
    }

    std::ifstream file(txtpath.c_str());
    return load_txt(file, features, nClasses);
}

FastForest fastforest::load_txt(std::istream& file, std::vector<std::string>& features, int nClasses) {
    if (nClasses < 2) {
        throw std::runtime_error("Error in fastforest::load_txt : nClasses has to be at least two");
    }

    const std::string info = "constructing FastForest from istream: ";

    FastForest ff;
    ff.baseResponses_.resize(nClasses == 2 ? 1 : nClasses);

    int treesSkipped = 0;

    int nVariables = 0;
    std::map<std::string, int> varIndices;
    bool fixFeatures = false;

    if (!features.empty()) {
        fixFeatures = true;
        nVariables = features.size();
        for (int i = 0; i < nVariables; ++i) {
            varIndices[features[i]] = i;
        }
    }

    std::string line;

    fastforest::detail::IndexMap nodeIndices;
    fastforest::detail::IndexMap leafIndices;

    int nPreviousNodes = 0;
    int nPreviousLeaves = 0;

    while (std::getline(file, line)) {
        std::size_t foundBegin = line.find("[");
        std::size_t foundEnd = line.find("]");
        if (foundBegin != std::string::npos) {
            std::string subline = line.substr(foundBegin + 1, foundEnd - foundBegin - 1);
            if (util::isInteger(subline) && !ff.responses_.empty()) {
                terminateTree(ff, nPreviousNodes, nPreviousLeaves, nodeIndices, leafIndices, treesSkipped);
            } else if (!util::isInteger(subline)) {
                std::stringstream ss(line);
                int index;
                ss >> index;
                line = ss.str();

                std::vector<std::string> splitstring = util::split(subline, '<');
                std::string const& varName = splitstring[0];
                FeatureType cutValue;
                {
                    std::stringstream ss(splitstring[1]);
                    ss >> cutValue;
                }
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
                util::NumericAfterSubstrOutput<int> output = util::numericAfterSubstr<int>(line, "yes=");
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

                ff.cutValues_.push_back(cutValue);
                ff.cutIndices_.push_back(varIndices[varName]);
                ff.leftIndices_.push_back(yes);
                ff.rightIndices_.push_back(no);
                std::size_t nNodeIndices = nodeIndices.size();
                nodeIndices[index] = nNodeIndices + nPreviousNodes;
            }

        } else {
            util::NumericAfterSubstrOutput<TreeResponseType> output =
                util::numericAfterSubstr<TreeResponseType>(line, "leaf=");
            if (output.found) {
                std::stringstream ss(line);
                int index;
                ss >> index;
                line = ss.str();

                ff.responses_.push_back(output.value);
                std::size_t nLeafIndices = leafIndices.size();
                leafIndices[index] = nLeafIndices + nPreviousLeaves;
            }
        }
    }
    terminateTree(ff, nPreviousNodes, nPreviousLeaves, nodeIndices, leafIndices, treesSkipped);

    if (nClasses > 2 && (ff.rootIndices_.size() + treesSkipped) % nClasses != 0) {
        std::stringstream ss;
        ss << "Error in FastForest construction : Forest has " << ff.rootIndices_.size()
           << " trees, which is not compatible with " << nClasses << "classes!";
        throw std::runtime_error(ss.str());
    }

    return ff;
}
