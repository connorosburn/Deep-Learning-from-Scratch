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
        std::vector<DataPair> trainingData();
        std::vector<DataPair> testData();

    private:
        std::vector<DataPair> loadData(std::string filename);
        DataPair readRow(std::string& row);
        std::vector<DataPair> training;
        std::vector<DataPair> test;
};

#endif