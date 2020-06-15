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

#include "common_details.h"

#include <vector>
#include <unordered_map>
#include <stdexcept>

        void fastforest::detail::correctIndices(std::vector<int>::iterator begin,
                            std::vector<int>::iterator end,
                            std::unordered_map<int, int> const& nodeIndices,
                            std::unordered_map<int, int> const& leafIndices) {
            for (auto it = begin; it != end; ++it) {
                if (nodeIndices.count(*it)) {
                    *it = nodeIndices.at(*it);
                } else if (leafIndices.count(*it)) {
                    *it = -leafIndices.at(*it);
                } else {
                    throw std::runtime_error("something is wrong in the node structure");
                }
            }
        }
