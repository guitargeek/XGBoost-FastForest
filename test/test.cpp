#define BOOST_TEST_MODULE fastforestTests
#include <boost/test/unit_test.hpp>

#include "fastforest.h"

#include <fstream>
#include <cmath>

constexpr FastForest::FeatureType tolerance = 1e-4;

BOOST_AUTO_TEST_CASE(ExampleTest) {
    std::vector<std::string> features{"f0", "f1", "f2", "f3", "f4"};

    FastForest fastForest("continuous/model.txt", features);

    std::vector<FastForest::FeatureType> input{0.0, 0.2, 0.4, 0.6, 0.8};

    FastForest::FeatureType score = fastForest(input.data());
    FastForest::FeatureType logistcScore = 1. / (1. + std::exp(-score));
}

BOOST_AUTO_TEST_CASE(BasicTest) {
    std::vector<std::string> features{"f0", "f1", "f2", "f3", "f4"};

    FastForest fastForest("continuous/model.txt", features);

    std::ifstream fileX("continuous/X.csv");
    std::ifstream filePreds("continuous/preds.csv");

    std::vector<FastForest::FeatureType> input(5);
    FastForest::FeatureType score;
    FastForest::FeatureType ref;

    for (int i = 0; i < 100; ++i) {
        for (auto& x : input) {
            fileX >> x;
        }
        score = fastForest(input.data());
        filePreds >> ref;

        BOOST_CHECK_CLOSE(score, ref, tolerance);
    }
}

BOOST_AUTO_TEST_CASE(SerializationTest) {
    {
        std::vector<std::string> features{"f0", "f1", "f2", "f3", "f4"};
        FastForest fastForest("continuous/model.txt", features);
        fastForest.save("continuous/forest.bin");
    }

    FastForest fastForest("continuous/forest.bin");

    std::ifstream fileX("continuous/X.csv");
    std::ifstream filePreds("continuous/preds.csv");

    std::vector<FastForest::FeatureType> input(5);
    FastForest::FeatureType score;
    FastForest::FeatureType ref;

    for (int i = 0; i < 100; ++i) {
        for (auto& x : input) {
            fileX >> x;
        }
        score = fastForest(input.data());
        filePreds >> ref;

        BOOST_CHECK_CLOSE(score, ref, tolerance);
    }
}

BOOST_AUTO_TEST_CASE(DiscreteTest) {
    std::vector<std::string> features{"f0", "f1", "f2", "f3", "f4"};

    FastForest fastForest("discrete/model.txt", features);

    std::ifstream fileX("discrete/X.csv");
    std::ifstream filePreds("discrete/preds.csv");

    std::vector<FastForest::FeatureType> input(5);
    FastForest::FeatureType score;
    FastForest::FeatureType ref;

    for (int i = 0; i < 100; ++i) {
        for (auto& x : input) {
            fileX >> x;
        }
        score = fastForest(input.data());
        filePreds >> ref;

        BOOST_CHECK_CLOSE(score, ref, tolerance);
    }
}

BOOST_AUTO_TEST_CASE(ManyfeaturesTest) {
    std::vector<std::string> features{};
    for (int i = 0; i < 310; ++i) {
        features.push_back(std::string("f") + std::to_string(i));
    }

    FastForest fastForest("manyfeatures/model.txt", features);

    std::ifstream fileX("manyfeatures/X.csv");
    std::ifstream filePreds("manyfeatures/preds.csv");

    std::vector<FastForest::FeatureType> input(features.size());
    FastForest::FeatureType score;
    FastForest::FeatureType ref;

    for (int i = 0; i < 100; ++i) {
        for (auto& x : input) {
            fileX >> x;
        }
        score = fastForest(input.data());
        filePreds >> ref;

        BOOST_CHECK_CLOSE(score, ref, tolerance);
    }
}
