/**

MIT License

Copyright (c) 2020 Jonas Rembser

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

#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <streambuf>

namespace util {

    std::string readFile(const char* filename);

}

std::string util::readFile(const char* filename) {
    std::ifstream t(filename);
    std::string str;

    t.seekg(0, std::ios::end);
    str.reserve(t.tellg());
    t.seekg(0, std::ios::beg);

    str.assign((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());

    return str;
}

namespace tmva {
    class XMLAttributes {
      public:
        // If we set an attribute that is already set, this will do nothing and return false.
        // Therefore an attribute has repeated and we know a new node has started.
        bool set(std::string const& name, std::string const& value) {
            if (name == "itree")
                return setValue(itree_, std::stoi(value));
            if (name == "boostWeight")
                return setValue(boostWeight_, std::stod(value));
            if (name == "pos")
                return setValue(pos_, value[0]);
            if (name == "depth")
                return setValue(depth_, std::stoi(value));
            if (name == "IVar")
                return setValue(IVar_, std::stoi(value));
            if (name == "Cut")
                return setValue(Cut_, std::stod(value));
            if (name == "res")
                return setValue(res_, std::stod(value));
            if (name == "nType")
                return setValue(nType_, std::stoi(value));
            return true;
        }

        bool hasValue(std::string const& name) {
            if (name == "itree")
                return itree_.has_value();
            if (name == "boostWeight")
                return boostWeight_.has_value();
            if (name == "pos")
                return pos_.has_value();
            if (name == "depth")
                return depth_.has_value();
            if (name == "IVar")
                return IVar_.has_value();
            if (name == "Cut")
                return Cut_.has_value();
            if (name == "res")
                return res_.has_value();
            if (name == "nType")
                return nType_.has_value();
            return false;
        }

        auto const& itree() const { return itree_; };
        auto const& boostWeight() const { return boostWeight_; };
        auto const& pos() const { return pos_; };
        auto const& depth() const { return depth_; };
        auto const& IVar() const { return IVar_; };
        auto const& Cut() const { return Cut_; };
        auto const& res() const { return res_; };
        auto const& nType() const { return nType_; };

        void reset() {
            boostWeight_.reset();
            itree_.reset();
            pos_.reset();
            depth_.reset();
            IVar_.reset();
            Cut_.reset();
            res_.reset();
            nType_.reset();
        }

      private:
        template <class T>
        bool setValue(std::optional<T>& member, T const& value) {
            if (member.has_value()) {
                member = value;
                return false;
            }
            member = value;
            return true;
        }

        // from the tree root node node
        std::optional<double> boostWeight_ = std::nullopt;
        std::optional<int> itree_ = std::nullopt;
        std::optional<char> pos_ = std::nullopt;
        std::optional<int> depth_ = std::nullopt;
        std::optional<int> IVar_ = std::nullopt;
        std::optional<double> Cut_ = std::nullopt;
        std::optional<double> res_ = std::nullopt;
        std::optional<int> nType_ = std::nullopt;
    };

    struct BDTWithXMLAttributes {
        std::vector<double> boostWeights;
        std::vector<std::vector<XMLAttributes>> nodes;
    };

    BDTWithXMLAttributes readXMLFile(std::string const& filename);

    BDTWithXMLAttributes readXMLFile(std::string const& filename) {
        const std::string str = util::readFile(filename.c_str());

        std::size_t pos1 = 0;

        std::string name;
        std::string value;

        BDTWithXMLAttributes bdtXmlAttributes;

        std::vector<XMLAttributes>* currentTree = nullptr;

        XMLAttributes* attrs = nullptr;

        while ((pos1 = str.find('=', pos1)) != std::string::npos) {
            auto pos2 = str.rfind(' ', pos1) + 1;

            name = str.substr(pos2, pos1 - pos2);

            pos2 = pos1 + 2;
            pos1 = str.find('"', pos2);

            value = str.substr(pos2, pos1 - pos2);

            if (name == "boostWeight") {
                bdtXmlAttributes.boostWeights.push_back(std::stod(value));
            }

            if (name == "itree") {
                bdtXmlAttributes.nodes.emplace_back();
                currentTree = &bdtXmlAttributes.nodes.back();
                currentTree->emplace_back();
                attrs = &currentTree->back();
            }

            if (bdtXmlAttributes.nodes.empty())
                continue;

            if (attrs->hasValue(name)) {
                currentTree->emplace_back();
                attrs = &currentTree->back();
            }

            attrs->set(name, value);
        }

        if (bdtXmlAttributes.nodes.size() != bdtXmlAttributes.boostWeights.size()) {
            throw std::runtime_error("nodes size and bosstWeights size don't match");
        }

        return bdtXmlAttributes;
    }

}  // namespace tmva

using namespace fastforest;

namespace {

    struct SlowTreeNode {
        bool isLeaf = false;
        int depth = -1;
        int index = -1;
        int yes = -1;
        int no = -1;
        int missing = -1;
        int cutIndex = -1;
        double cutValue = 0.0;
        double leafValue = 0.0;
    };

    std::vector<SlowTreeNode> getSlowTreeNodes(std::vector<tmva::XMLAttributes> const& nodes) {
        std::vector<SlowTreeNode> xgbNodes(nodes.size());

        int xgbIndex = 0;
        for (int depth = 0; xgbIndex != nodes.size(); ++depth) {
            int iNode = 0;
            for (auto const& node : nodes) {
                if (node.depth() == depth) {
                    xgbNodes[iNode].index = xgbIndex;
                    ++xgbIndex;
                }
                ++iNode;
            }
        }

        int iNode = 0;
        for (auto const& node : nodes) {
            auto& xgbNode = xgbNodes[iNode];
            xgbNode.isLeaf = *node.nType() != 0;
            xgbNode.depth = *node.depth();
            xgbNode.cutIndex = *node.IVar();
            xgbNode.cutValue = *node.Cut();
            xgbNode.leafValue = *node.res();
            if (!xgbNode.isLeaf) {
                xgbNode.yes = xgbNodes[iNode + 1].index;
                xgbNode.no = xgbNode.yes + 1;
                xgbNode.missing = xgbNode.yes;
            }
            ++iNode;
        }

        return xgbNodes;
    }

    using SlowTree = std::vector<SlowTreeNode>;
    using SlowForest = std::vector<SlowTree>;

    std::ostream& operator<<(std::ostream& os, SlowTreeNode const& node) {
        for (int i = 0; i < node.depth; ++i) {
            os << "\t";
        }
        if (node.isLeaf) {
            os << node.index << ":leaf=" << node.leafValue;
        } else {
            os << node.index << ":[f" << node.cutIndex << "<" << node.cutValue << "]";
            os << " yes=" << node.yes << ",no=" << node.no << ",missing=" << node.missing;
        }
        return os;
    }

    std::ostream& operator<<(std::ostream& os, SlowTree const& nodes) {
        for (auto const& node : nodes) {
            os << node << "\n";
        }
        return os;
    }

    std::ostream& operator<<(std::ostream& os, SlowForest const& forest) {
        int iTree = 0;
        for (auto const& tree : forest) {
            os << "booster[" << iTree << "]:"
               << "\n";
            os << tree;
            ++iTree;
        }
        return os;
    }

}  // namespace

namespace fastforest {

    FastForest load_slowforest(SlowForest const& xgb, std::vector<std::string>& features) {
        FastForest ff;

        int nVariables = 0;
        std::unordered_map<std::string, int> varIndices;
        bool fixFeatures = false;

        std::unordered_map<int, int> nodeIndices;
        std::unordered_map<int, int> leafIndices;

        int nPreviousNodes = 0;
        int nPreviousLeaves = 0;

        for (auto const& tree : xgb) {
            detail::correctIndices(
                ff.rightIndices_.begin() + nPreviousNodes, ff.rightIndices_.end(), nodeIndices, leafIndices);
            detail::correctIndices(
                ff.leftIndices_.begin() + nPreviousNodes, ff.leftIndices_.end(), nodeIndices, leafIndices);
            nodeIndices.clear();
            leafIndices.clear();
            nPreviousNodes = ff.cutValues_.size();
            nPreviousLeaves = ff.responses_.size();
            ff.rootIndices_.push_back(nPreviousNodes);
            for (auto const& node : tree) {
                if (node.isLeaf) {
                    ff.responses_.push_back(node.leafValue);
                    leafIndices[node.index] = leafIndices.size() + nPreviousLeaves;
                } else {
                    ff.cutValues_.push_back(node.cutValue);
                    ff.cutIndices_.push_back(node.cutIndex);
                    ff.leftIndices_.push_back(node.yes);
                    ff.rightIndices_.push_back(node.no);
                    nodeIndices[node.index] = nodeIndices.size() + nPreviousNodes;
                }
            }
        }

        detail::correctIndices(
            ff.rightIndices_.begin() + nPreviousNodes, ff.rightIndices_.end(), nodeIndices, leafIndices);
        detail::correctIndices(
            ff.leftIndices_.begin() + nPreviousNodes, ff.leftIndices_.end(), nodeIndices, leafIndices);

        return ff;
    }

    FastForest load_tmva_xml(std::string const& xmlpath, std::vector<std::string>& features) {
        tmva::BDTWithXMLAttributes tmvaXML = tmva::readXMLFile(xmlpath);
        std::vector<std::vector<SlowTreeNode>> xgboostForest;
        for (auto const& tree : tmvaXML.nodes) {
            xgboostForest.push_back(getSlowTreeNodes(tree));
        }
        return fastforest::load_slowforest(xgboostForest, features);
    }
}  // namespace fastforest
