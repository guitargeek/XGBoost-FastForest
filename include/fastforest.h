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

class FastForest {
  public:
    FastForest(std::string const& txtpath, std::vector<std::string>& features);
    FastForest(std::string const& txtpath);
    double operator()(const float* array) const;

    auto const& cutIndices() const { return cutIndices_; }
    auto const& cutValues() const { return cutValues_; }
    auto const& leftIndices() const { return leftIndices_; }
    auto const& rightIndices() const { return rightIndices_; }
    auto const& responses() const { return responses_; }

    void save(std::string const& filename) const;

  private:
    std::vector<int> rootIndices_;
    std::vector<unsigned char> cutIndices_;
    std::vector<float> cutValues_;
    std::vector<int> leftIndices_;
    std::vector<int> rightIndices_;
    std::vector<float> responses_;
};

#endif
