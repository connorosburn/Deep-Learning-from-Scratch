#ifndef MNIST_LOADER_HPP
#define MNIST_LOADER_HPP

#include <string>
#include <vector>

struct DataPair {
    DataPair(int imageLabel, std::vector<std::vector<double>> imageData): label(imageLabel), image(imageData) {};
    int label;
    std::vector<std::vector<double>> image;
};

class MNISTLoader {
    public:
        const std::vector<DataPair>& trainingData();
        const std::vector<DataPair>& testData();

    private:
        std::vector<DataPair> loadData(std::string imageFileName, std::string labelFileName);
        std::vector<DataPair> training;
        std::vector<DataPair> test;
};

#endif