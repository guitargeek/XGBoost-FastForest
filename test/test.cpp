#define BOOST_TEST_MODULE fastforestTests
#include <boost/test/unit_test.hpp>

#include "fastforest.h"

#include <fstream>
#include <cmath>

constexpr fastforest::FeatureType tolerance = 1e-4;

BOOST_AUTO_TEST_CASE(ExampleTest) {
    std::vector<std::string> features{"f0", "f1", "f2", "f3", "f4"};

    const auto fastForest = fastforest::load_txt("continuous/model.txt", features);

    std::vector<fastforest::FeatureType> input{0.0, 0.2, 0.4, 0.6, 0.8};

    fastforest::FeatureType score = fastForest(input.data());
    fastforest::FeatureType logistcScore = 1. / (1. + std::exp(-score));
}

BOOST_AUTO_TEST_CASE(BasicTest) {
    std::vector<std::string> features{"f0", "f1", "f2", "f3", "f4"};

    const auto fastForest = fastforest::load_txt("continuous/model.txt", features);

    std::ifstream fileX("continuous/X.csv");
    std::ifstream filePreds("continuous/preds.csv");

    std::vector<fastforest::FeatureType> input(5);
    fastforest::FeatureType score;
    double ref;

    for (int i = 0; i < 100; ++i) {
        for (auto& x : input) {
            fileX >> x;
        }
        score = fastForest(input.data());
        filePreds >> ref;

        BOOST_CHECK_CLOSE(score, ref, tolerance);
    }
}

BOOST_AUTO_TEST_CASE(SoftmaxTest) {
    std::vector<std::string> features{"f0", "f1", "f2", "f3", "f4"};

    const auto fastForest = fastforest::load_txt("softmax/model.txt", features);

    std::ifstream fileX("softmax/X.csv");
    std::ifstream filePreds("softmax/preds.csv");

    std::vector<fastforest::FeatureType> input(5);
    fastforest::FeatureType score;
    double ref;

    for (int i = 0; i < 100; ++i) {
        for (auto& x : input) {
            fileX >> x;
        }
        for (auto& x : fastForest.softmax(input.data(), 3)) {
            filePreds >> ref;
            BOOST_CHECK_CLOSE(x, ref, tolerance);
        }
    }
}

BOOST_AUTO_TEST_CASE(SoftmaxArrayTest) {
    std::vector<std::string> features{"f0", "f1", "f2", "f3", "f4"};

    const auto fastForest = fastforest::load_txt("softmax/model.txt", features);

    std::ifstream fileX("softmax/X.csv");
    std::ifstream filePreds("softmax/preds.csv");

    std::vector<fastforest::FeatureType> input(5);
    fastforest::FeatureType score;
    double ref;

    for (int i = 0; i < 100; ++i) {
        for (auto& x : input) {
            fileX >> x;
        }
        for (auto& x : fastForest.softmax<3>(input.data())) {
            filePreds >> ref;
            BOOST_CHECK_CLOSE(x, ref, tolerance);
        }
    }
}

BOOST_AUTO_TEST_CASE(SerializationTest) {
    {
        std::vector<std::string> features{"f0", "f1", "f2", "f3", "f4"};
        const auto fastForest = fastforest::load_txt("continuous/model.txt", features);
        fastForest.write_bin("continuous/forest.bin");
    }

    const auto fastForest = fastforest::load_bin("continuous/forest.bin");

    std::ifstream fileX("continuous/X.csv");
    std::ifstream filePreds("continuous/preds.csv");

    std::vector<fastforest::FeatureType> input(5);
    fastforest::FeatureType score;
    double ref;

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

    const auto fastForest = fastforest::load_txt("discrete/model.txt", features);

    std::ifstream fileX("discrete/X.csv");
    std::ifstream filePreds("discrete/preds.csv");

    std::vector<fastforest::FeatureType> input(5);
    fastforest::FeatureType score;
    double ref;

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
    for (int i = 0; i < 311; ++i) {
        features.push_back(std::string("f") + std::to_string(i));
    }

    const auto fastForest = fastforest::load_txt("manyfeatures/model.txt", features);

    std::ifstream fileX("manyfeatures/X.csv");
    std::ifstream filePreds("manyfeatures/preds.csv");

    std::vector<fastforest::FeatureType> input(features.size());
    fastforest::FeatureType score;
    double ref;

    for (int i = 0; i < 100; ++i) {
        for (auto& x : input) {
            fileX >> x;
        }
        score = fastForest(input.data());
        filePreds >> ref;

        BOOST_CHECK_CLOSE(score, ref, tolerance);
    }
}

#ifdef EXPERIMENTAL_TMVA_SUPPORT

BOOST_AUTO_TEST_CASE(BasicTMVAXMLTest) {
    std::vector<std::string> features{"f0", "f1", "f2", "f3", "f4"};

    const auto fastForest = fastforest::load_tmva_xml("continuous/model.xml", features);

    std::ifstream fileX("continuous/X.csv");
    std::ifstream filePreds("continuous/preds.csv");

    std::vector<fastforest::FeatureType> input(5);
    fastforest::FeatureType score;
    double ref;

    for (int i = 0; i < 100; ++i) {
        for (auto& x : input) {
            fileX >> x;
        }
        score = fastForest(input.data());
        filePreds >> ref;

        BOOST_CHECK_CLOSE(score, ref, tolerance);
    }
}

#endif
