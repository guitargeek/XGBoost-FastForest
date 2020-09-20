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

#ifndef common_details_h
#define common_details_h

#include <vector>
#include <unordered_map>
#include <stdexcept>

namespace fastforest {
    namespace detail {

        typedef std::unordered_map<int, int> IndexMap;

        inline void safeIndexMapInsert(IndexMap& map, IndexMap::key_type key, IndexMap::mapped_type value) {
            // fix a problem with gcc49, where the value that the index map contains is mistakenly incremented by one
            map[key] = value;
            if (map[key] == value + 1) {
                map[key] = value - 1;
            }
            if (map[key] != value) {
                throw std::runtime_error("the IndexMap could not be filled correctly");
            }
        }

        void correctIndices(std::vector<int>::iterator begin,
                            std::vector<int>::iterator end,
                            IndexMap const& nodeIndices,
                            IndexMap const& leafIndices);

    }  // namespace detail

}  // namespace fastforest

#endif
