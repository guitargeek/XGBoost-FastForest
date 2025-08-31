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

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <streambuf>
#include <map>

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
        XMLAttributes() {
            boostWeight_ = NULL;
            itree_ = NULL;
            pos_ = NULL;
            depth_ = NULL;
            IVar_ = NULL;
            Cut_ = NULL;
            res_ = NULL;
            nType_ = NULL;
        };

        XMLAttributes(XMLAttributes const& other) {
            boostWeight_ = NULL;
            itree_ = NULL;
            pos_ = NULL;
            depth_ = NULL;
            IVar_ = NULL;
            Cut_ = NULL;
            res_ = NULL;
            nType_ = NULL;
            *this = other;
        };

        XMLAttributes& operator=(XMLAttributes const& other) {
            if (other.boostWeight_ != NULL) {
                boostWeight_ = new double;
                *boostWeight_ = *other.boostWeight_;
            }
            if (other.itree_ != NULL) {
                itree_ = new int;
                *itree_ = *other.itree_;
            }
            if (other.pos_ != NULL) {
                pos_ = new char;
                *pos_ = *other.pos_;
            }
            if (other.depth_ != NULL) {
                depth_ = new int;
                *depth_ = *other.depth_;
            }
            if (other.IVar_ != NULL) {
                IVar_ = new int;
                *IVar_ = *other.IVar_;
            }
            if (other.Cut_ != NULL) {
                Cut_ = new double;
                *Cut_ = *other.Cut_;
            }
            if (other.res_ != NULL) {
                res_ = new double;
                *res_ = *other.res_;
            }
            if (other.nType_ != NULL) {
                nType_ = new int;
                *nType_ = *other.nType_;
            }
            return *this;
        }

        ~XMLAttributes() { reset(); }

        // If we set an attribute that is already set, this will do nothing and return false.
        // Therefore an attribute has repeated and we know a new node has started.
        void set(std::string const& name, std::string const& value) {
            if (name == "itree")
                return setValue(itree_, value);
            if (name == "boostWeight")
                return setValue(boostWeight_, value);
            if (name == "pos")
                return setValue(pos_, value.substr(0, 1));
            if (name == "depth")
                return setValue(depth_, value);
            if (name == "IVar")
                return setValue(IVar_, value);
            if (name == "Cut")
                return setValue(Cut_, value);
            if (name == "res")
                return setValue(res_, value);
            if (name == "nType")
                return setValue(nType_, value);
        }

        bool hasValue(std::string const& name) {
            if (name == "itree")
                return itree_ != NULL;
            if (name == "boostWeight")
                return boostWeight_ != NULL;
            if (name == "pos")
                return pos_ != NULL;
            if (name == "depth")
                return depth_ != NULL;
            if (name == "IVar")
                return IVar_ != NULL;
            if (name == "Cut")
                return Cut_ != NULL;
            if (name == "res")
                return res_ != NULL;
            if (name == "nType")
                return nType_ != NULL;
            return false;
        }

        int const* itree() const { return itree_; };
        double const* boostWeight() const { return boostWeight_; };
        char const* pos() const { return pos_; };
        int const* depth() const { return depth_; };
        int const* IVar() const { return IVar_; };
        double const* Cut() const { return Cut_; };
        double const* res() const { return res_; };
        int const* nType() const { return nType_; };

        void reset() {
            if (boostWeight_ != NULL) {
                delete boostWeight_;
                boostWeight_ = NULL;
            }
            if (itree_ != NULL) {
                delete itree_;
                itree_ = NULL;
            }
            if (pos_ != NULL) {
                delete pos_;
                pos_ = NULL;
            }
            if (depth_ != NULL) {
                delete depth_;
                depth_ = NULL;
            }
            if (IVar_ != NULL) {
                delete IVar_;
                IVar_ = NULL;
            }
            if (Cut_ != NULL) {
                delete Cut_;
                Cut_ = NULL;
            }
            if (res_ != NULL) {
                delete res_;
                res_ = NULL;
            }
            if (nType_ != NULL) {
                delete nType_;
                nType_ = NULL;
            }
        }

      private:
        template <class T>
        void setValue(T*& member, std::string const& value) {
            if (!member) {
                member = new T;
            }
            std::stringstream ss(value);
            ss >> *member;
        }

        // from the tree root node node
        double* boostWeight_;
        int* itree_;
        char* pos_;
        int* depth_;
        int* IVar_;
        double* Cut_;
        double* res_;
        int* nType_;
    };

    struct BDTWithXMLAttributes {
        std::vector<double> boostWeights;
        std::vector<std::vector<XMLAttributes> > nodes;
    };

    BDTWithXMLAttributes readXMLFile(std::string const& filename);

    BDTWithXMLAttributes readXMLFile(std::string const& filename) {
        const std::string str = util::readFile(filename.c_str());

        std::size_t pos1 = 0;

        std::string name;
        std::string value;

        BDTWithXMLAttributes bdtXmlAttributes;

        std::vector<XMLAttributes>* currentTree = NULL;

        XMLAttributes* attrs = NULL;

        while ((pos1 = str.find('=', pos1)) != std::string::npos) {
            std::size_t pos2 = str.rfind(' ', pos1) + 1;

            name = str.substr(pos2, pos1 - pos2);

            pos2 = pos1 + 2;
            pos1 = str.find('"', pos2);

            value = str.substr(pos2, pos1 - pos2);

            if (name == "boostWeight") {
                bdtXmlAttributes.boostWeights.push_back(0.0);
                std::stringstream(value) >> bdtXmlAttributes.boostWeights.back();
            }

            if (name == "itree") {
                bdtXmlAttributes.nodes.push_back(std::vector<tmva::XMLAttributes>());
                currentTree = &bdtXmlAttributes.nodes.back();
                currentTree->push_back(tmva::XMLAttributes());
                attrs = &currentTree->back();
            }

            if (bdtXmlAttributes.nodes.empty())
                continue;

            if (attrs->hasValue(name)) {
                currentTree->push_back(tmva::XMLAttributes());
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
        SlowTreeNode() {
            isLeaf = false;
            depth = -1;
            index = -1;
            yes = -1;
            no = -1;
            missing = -1;
            cutIndex = -1;
            cutValue = 0.0;
            leafValue = 0.0;
        }
        bool isLeaf;
        int depth;
        int index;
        int yes;
        int no;
        int missing;
        int cutIndex;
        double cutValue;
        double leafValue;
    };

    std::vector<SlowTreeNode> getSlowTreeNodes(std::vector<tmva::XMLAttributes> const& nodes) {
        std::vector<SlowTreeNode> xgbNodes(nodes.size());

        int xgbIndex = 0;
        for (int depth = 0; xgbIndex != nodes.size(); ++depth) {
            int iNode = 0;
            for (std::vector<tmva::XMLAttributes>::const_iterator node = nodes.begin(); node != nodes.end(); ++node) {
                if (*node->depth() == depth) {
                    xgbNodes[iNode].index = xgbIndex;
                    ++xgbIndex;
                }
                ++iNode;
            }
        }

        int iNode = 0;
        for (std::vector<tmva::XMLAttributes>::const_iterator node = nodes.begin(); node != nodes.end(); ++node) {
            SlowTreeNode& xgbNode = xgbNodes[iNode];
            xgbNode.isLeaf = *node->nType() != 0;
            xgbNode.depth = *node->depth();
            xgbNode.cutIndex = *node->IVar();
            xgbNode.cutValue = *node->Cut();
            xgbNode.leafValue = *node->res();
            if (!xgbNode.isLeaf) {
                xgbNode.yes = xgbNodes[iNode + 1].index;
                xgbNode.no = xgbNode.yes + 1;
                xgbNode.missing = xgbNode.yes;
            }
            ++iNode;
        }

        return xgbNodes;
    }

    typedef std::vector<SlowTreeNode> SlowTree;
    typedef std::vector<SlowTree> SlowForest;

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
        for (SlowTree::const_iterator node = nodes.begin(); node != nodes.end(); ++node) {
            os << *node << "\n";
        }
        return os;
    }

    std::ostream& operator<<(std::ostream& os, SlowForest const& forest) {
        int iTree = 0;
        for (SlowForest::const_iterator tree = forest.begin(); tree != forest.end(); ++tree) {
            os << "booster[" << iTree << "]:"
               << "\n";
            os << *tree;
            ++iTree;
        }
        return os;
    }

}  // namespace

namespace fastforest {

    FastForest load_slowforest(SlowForest const& xgb, std::vector<std::string>& features) {
        FastForest ff;
        ff.baseResponses_.resize(2);

        int nVariables = 0;
        std::map<std::string, int> varIndices;
        bool fixFeatures = false;

        std::map<int, int> nodeIndices;
        std::map<int, int> leafIndices;

        int nPreviousNodes = 0;
        int nPreviousLeaves = 0;

        for (SlowForest::const_iterator tree = xgb.begin(); tree != xgb.end(); ++tree) {
            detail::correctIndices(
                ff.rightIndices_.begin() + nPreviousNodes, ff.rightIndices_.end(), nodeIndices, leafIndices);
            detail::correctIndices(
                ff.leftIndices_.begin() + nPreviousNodes, ff.leftIndices_.end(), nodeIndices, leafIndices);
            nodeIndices.clear();
            leafIndices.clear();
            nPreviousNodes = ff.cutValues_.size();
            nPreviousLeaves = ff.responses_.size();
            ff.rootIndices_.push_back(nPreviousNodes);
            for (SlowTree::const_iterator node = tree->begin(); node != tree->end(); ++node) {
                if (node->isLeaf) {
                    ff.responses_.push_back(node->leafValue);
                    leafIndices[node->index] = leafIndices.size() + nPreviousLeaves;
                } else {
                    ff.cutValues_.push_back(node->cutValue);
                    ff.cutIndices_.push_back(node->cutIndex);
                    ff.leftIndices_.push_back(node->yes);
                    ff.rightIndices_.push_back(node->no);
                    nodeIndices[node->index] = nodeIndices.size() + nPreviousNodes;
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
        std::vector<std::vector<SlowTreeNode> > xgboostForest;
        for (std::vector<std::vector<tmva::XMLAttributes> >::const_iterator tree = tmvaXML.nodes.begin();
             tree != tmvaXML.nodes.end();
             ++tree) {
            xgboostForest.push_back(getSlowTreeNodes(*tree));
        }
        return fastforest::load_slowforest(xgboostForest, features);
    }
}  // namespace fastforest
