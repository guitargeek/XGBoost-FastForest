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
#include "common_details.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <stdexcept>

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
            explicit NumericAfterSubstrOutput() : value{0}, found{false}, failed{true} {}
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

        bool exists(std::string const& filename) {
            if (FILE *file = fopen(filename.c_str(), "r")) {
                fclose(file);
                return true;
            } else {
                return false;
            }   
        }

    }  // namespace util

}  // namespace

FastForest fastforest::load_txt(std::string const& txtpath, std::vector<std::string>& features) {
    const std::string info = "constructing FastForest from " + txtpath + ": ";

    if (!util::exists(txtpath)) {
        throw std::runtime_error(info + "file does not exists");
    }

    FastForest ff;

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

    int nPreviousNodes = 0;
    int nPreviousLeaves = 0;

    while (std::getline(file, line)) {
        auto foundBegin = line.find("[");
        auto foundEnd = line.find("]");
        if (foundBegin != std::string::npos) {
            auto subline = line.substr(foundBegin + 1, foundEnd - foundBegin - 1);
            if (util::isInteger(subline)) {
                detail::correctIndices(
                    ff.rightIndices_.begin() + nPreviousNodes, ff.rightIndices_.end(), nodeIndices, leafIndices);
                detail::correctIndices(
                    ff.leftIndices_.begin() + nPreviousNodes, ff.leftIndices_.end(), nodeIndices, leafIndices);
                nodeIndices.clear();
                leafIndices.clear();
                nPreviousNodes = ff.cutValues_.size();
                nPreviousLeaves = ff.responses_.size();
                ff.rootIndices_.push_back(nPreviousNodes);
            } else {
                std::stringstream ss(line);
                int index;
                ss >> index;
                line = ss.str();

                auto splitstring = util::split(subline, '<');
                auto const& varName = splitstring[0];
                FeatureType cutValue = std::stold(splitstring[1]);
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

                ff.cutValues_.push_back(cutValue);
                ff.cutIndices_.push_back(varIndices[varName]);
                ff.leftIndices_.push_back(yes);
                ff.rightIndices_.push_back(no);
                nodeIndices[index] = nodeIndices.size() + nPreviousNodes;
            }

        } else {
            auto output = util::numericAfterSubstr<TreeResponseType>(line, "leaf=");
            if (output.found) {
                std::stringstream ss(line);
                int index;
                ss >> index;
                line = ss.str();

                ff.responses_.push_back(output.value);
                leafIndices[index] = leafIndices.size() + nPreviousLeaves;
            }
        }
    }
    detail::correctIndices(ff.rightIndices_.begin() + nPreviousNodes, ff.rightIndices_.end(), nodeIndices, leafIndices);
    detail::correctIndices(ff.leftIndices_.begin() + nPreviousNodes, ff.leftIndices_.end(), nodeIndices, leafIndices);

    return ff;
}
