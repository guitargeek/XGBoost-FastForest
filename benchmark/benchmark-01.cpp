// compile with g++ -o benchmark-01 benchmark-01.cpp -lfastforest
//
// optimization flag does not matter because fastforest is already compiled

#include "fastforest.h"

#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>
#include <iostream>
#include <ctime>

int main() {
    std::vector<std::string> features{"f0", "f1", "f2", "f3", "f4"};

    const auto fastForest = fastforest::load_txt("model.txt", features);

    const int n = 100000;

    std::vector<float> input(5 * n);
    std::vector<double> scores(n);

    std::generate(input.begin(), input.end(), std::rand);
    for (auto& x : input) {
        x = float(x) / RAND_MAX * 10 - 5;
    }

    clock_t begin = clock();
    for (int i = 0; i < n; ++i) {
        scores[i] = 1. / (1. + std::exp(-fastForest(input.data() + i * 5)));
    }
    double average = std::accumulate(scores.begin(), scores.end(), 0.0) / scores.size();
    std::cout << average << std::endl;

    clock_t end = clock();
    double elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;

    std::cout << "Wall time for inference: " << elapsedSecs << " s" << std::endl;
}
