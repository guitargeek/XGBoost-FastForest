#define BOOST_TEST_MODULE fastforestTests
#include <boost/test/unit_test.hpp>

#include "fastforest.h"

#include <fstream>
#include <cmath>

BOOST_AUTO_TEST_CASE(ExampleTest) {
    std::vector<std::string> features{"f0", "f1", "f2", "f3", "f4"};

    FastForest fastForest("model.txt", features);

    std::vector<float> input{0.0, 0.2, 0.4, 0.6, 0.8};

    float score = fastForest(input.data());
    float logistcScore = 1. / (1. + std::exp(-score));
}

BOOST_AUTO_TEST_CASE(BasicTest) {
    std::vector<std::string> features{"f0", "f1", "f2", "f3", "f4"};

    FastForest fastForest("model.txt", features);

    std::ifstream fileX("X.csv");
    std::ifstream filePreds("preds.csv");

    std::vector<float> input(5);
    float score;
    float ref;

    for (int i = 0; i < 100; ++i) {
        for (auto& x : input) {
            fileX >> x;
        }
        score = fastForest(input.data());
        filePreds >> ref;
    }

    BOOST_CHECK_CLOSE(score, ref, 0.001);
}

BOOST_AUTO_TEST_CASE(SerializationTest) {
    {
        std::vector<std::string> features{"f0", "f1", "f2", "f3", "f4"};
        FastForest fastForest("model.txt", features);
        fastForest.save("forest.bin");
    }

    FastForest fastForest("forest.bin");

    std::ifstream fileX("X.csv");
    std::ifstream filePreds("preds.csv");

    std::vector<float> input(5);
    float score;
    float ref;

    for (int i = 0; i < 100; ++i) {
        for (auto& x : input) {
            fileX >> x;
        }
        score = fastForest(input.data());
        filePreds >> ref;
    }

    BOOST_CHECK_CLOSE(score, ref, 0.001);
}
