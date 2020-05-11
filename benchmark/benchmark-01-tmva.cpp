// compile with g++ -o benchmark-01-tmva benchmark-01-tmva.cpp `root-config --cflags --glibs` -lTMVA
//
// optimization flag does not matter because the TMVA library is already compiled

#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>
#include <iostream>
#include <ctime>

#include "TMVA/Reader.h"

int main() {
    TMVA::Reader reader("!Color:!Silent");

    float f0;
    float f1;
    float f2;
    float f3;
    float f4;

    reader.AddVariable("f0", &f0);
    reader.AddVariable("f1", &f1);
    reader.AddVariable("f2", &f2);
    reader.AddVariable("f3", &f3);
    reader.AddVariable("f4", &f4);

    reader.BookMVA("BDTG", "model.xml");

    const int n = 100000;

    std::vector<float> input(5 * n);
    std::vector<double> scores(n);

    std::generate(input.begin(), input.end(), std::rand);
    for (auto& x : input) {
        x = float(x) / RAND_MAX * 10 - 5;
    }

    clock_t begin = clock();
    for (int i = 0; i < n; ++i) {
        f0 = input[i * 5];
        f1 = input[i * 5 + 1];
        f2 = input[i * 5 + 2];
        f3 = input[i * 5 + 3];
        f4 = input[i * 5 + 4];
        scores[i] = reader.EvaluateMVA("BDTG");
    }
    double average = std::accumulate(scores.begin(), scores.end(), 0.0) / scores.size();
    std::cout << average << std::endl;

    clock_t end = clock();
    double elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;

    std::cout << "Wall time for inference: " << elapsedSecs << " s" << std::endl;
}
