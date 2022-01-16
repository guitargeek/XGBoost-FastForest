// compile with g++ -o benchmark-01-m2c benchmark-01-m2c.cpp
//
// as the model is compiled together with this test, please try out different optimization flags
// and quote the best result for a fair comparison with fastforest

// the model.c file assumes hardcoded feature indices
#define f0 0
#define f1 1
#define f2 2
#define f3 3
#define f4 4
#include "model.c"

#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>
#include <iostream>
#include <ctime>

int main() {
    const int n = 100000;

    std::vector<float> input(5 * n);
    std::vector<float> scores(n);

    std::generate(input.begin(), input.end(), std::rand);
    for (auto& x : input) {
        x = float(x) / RAND_MAX * 10 - 5;
    }

    clock_t begin = clock();
    float out;
    for (int i = 0; i < n; ++i) {
        score(input.data() + i * 5, &scores[i]);
    }
    double average = std::accumulate(scores.begin(), scores.end(), 0.0) / scores.size();
    std::cout << average << std::endl;

    clock_t end = clock();
    double elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;

    std::cout << "Wall time for inference: " << elapsedSecs << " s" << std::endl;
}
