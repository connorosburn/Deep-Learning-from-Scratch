#include "MNISTLoader.hpp"
#include <fstream>
#include <iostream>

std::vector<DataPair> MNISTLoader::trainingData() {
    if(training.empty()) {
        training = loadData("lib/MNISTFashion/fashion-mnist_train.csv");
    }
    return training;
}

std::vector<DataPair> MNISTLoader::testData() {
    if(test.empty()) {
        test = loadData("lib/MNISTFashion/fashion-mnist_test.csv");
    }
    return test;
}

std::vector<DataPair> MNISTLoader::loadData(std::string filename) {
    std::ifstream file;
    file.open(filename);
    std::string output;

    // skips over the first line with the CSV labels
    getline(file, output);

    std::vector<DataPair> data;
    while(!file.eof()) {
        getline(file, output);
        if(!output.empty()) {
            data.push_back(readRow(output));
        }
    }
    file.close();
    return data;
}

int nextComma(std::string& row) {
    auto firstComma = row.find_first_of(",");
    std::string output;
    if(firstComma > row.size()) {
        output = row;
        row.clear();
    } else {
        output = row.substr(0, firstComma);
        row.erase(0, firstComma + 1);
    }
    return std::stoi(output);
}

DataPair MNISTLoader::readRow(std::string& row) {
    int label = nextComma(row);
    std::vector<std::vector<double>> image;
    const int ROW_LENGTH = 28;
    while(!row.empty()) {
        image.emplace_back();
        for(int i = 0; i < ROW_LENGTH; i++) {
            double pixel = double(nextComma(row)) / double(255);
            image.back().push_back(pixel);
        }
    }
    return DataPair(label, image);
}