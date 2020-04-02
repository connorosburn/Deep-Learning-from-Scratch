#include "MNISTLoader.hpp"
#include <fstream>
#include <iostream>

std::vector<DataPair> MNISTLoader::trainingData() {
    if(training.empty()) {
        training = loadData("lib/MNISTFashion/train-images-idx3-ubyte", "lib/MNISTFashion/train-labels-idx1-ubyte");
    }
    return training;
}

std::vector<DataPair> MNISTLoader::testData() {
    if(test.empty()) {
        test = loadData("lib/MNISTFashion/t10k-images-idx3-ubyte", "lib/MNISTFashion/t10k-labels-idx1-ubyte");
    }
    return test;
}

std::vector<DataPair> MNISTLoader::loadData(std::string imageFileName, std::string labelFileName) {
    int imageHolder;
    std::ifstream imageFile;
    imageFile.open(imageFileName, std::ios::binary | std::ios::in);
    //reads past magic number
    imageFile.read((char*)&imageHolder, 4);
    //reads the number of examples
    imageFile.read((char*)&imageHolder, 4);

    int labelHolder;
    std::ifstream labelFile;
    labelFile.open(labelFileName, std::ios::binary | std::ios::in);
    //reads past magic number
    labelFile.read((char*)&labelHolder, 4);
    //reads the number of examples
    labelFile.read((char*)&labelHolder, 4);

    int imageCount = ntohl(imageHolder);
    int labelCount = ntohl(labelHolder);

    if(imageCount != labelCount) {
        throw "image and label counts must be the same";
    }

    //skips past the row and column counts
    imageFile.read((char*)&imageHolder, 8);

    std::vector<DataPair> data;
    for(int i  = 0; i < imageCount; i++) {
        std::vector<std::vector<double>> image;
        for(int y = 0;  y < 28; y++) {
            image.emplace_back();
            for(int x = 0; x < 28; x++) {
                unsigned int pixel;
                imageFile.read((char*)&pixel, 1);
                image.back().emplace_back(double(pixel) / double(255));
            }
        }
        unsigned int label;
        labelFile.read((char*)&label, 1);
        data.emplace_back(label, image);
    }
    return data;
}