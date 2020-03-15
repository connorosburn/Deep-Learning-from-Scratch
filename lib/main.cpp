#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include "NeuralNetwork/Layer/FullyConnectedLayer.hpp"
#include <functional>
#include <vector>
#include <typeinfo>

std::vector<double> randomBinary(int length) {
    //totes copied/pasted (and modified), throwaway code anyway
    std::vector<double> output;
    static auto dev = std::random_device();
    static auto gen = std::mt19937{dev()};
    static auto dist = std::uniform_real_distribution<double>(0,1);
    for(int i = 0; i < length; i++) {
        if(dist(gen) < 0.5) {
            output.push_back(0);
        } else {
            output.push_back(1);
        }
    }
    return output;
}

double baseTenFromBinary(std::vector<double> binaryDigits) {
    double sum = 0;
    for(int i = 0; i < binaryDigits.size(); i++) {
        int position = binaryDigits.size() - 1 - i;
        sum += binaryDigits[position] * std::pow(2, i);
    }
    return sum;
}

double binarySum(std::vector<double> binaryDigits) {
    auto halfIter = binaryDigits.begin() + (binaryDigits.size() / 2);
    std::vector<double> firstBinary = std::vector<double>(binaryDigits.begin(), halfIter);
    std::vector<double> secondBinary = std::vector<double>(halfIter, binaryDigits.end());
    return baseTenFromBinary(firstBinary) + baseTenFromBinary(secondBinary);
}

struct BinaryData {
    BinaryData() {
        binaryDigits = randomBinary(8);
        baseTenSum = binarySum(binaryDigits);
    }
    std::vector<double> binaryDigits;
    double baseTenSum;
};

int main() {
    const double LEARNING_RATE = 0.001;

    std::vector<double> input(8, 0);
    std::vector<double> totalError(8, 0);
    FullyConnectedLayer hiddenLayer({input.begin(), input.end()}, {totalError.begin(), totalError.end()}, 16, Activation::relu);
    FullyConnectedLayer hiddenLayer2(hiddenLayer.getOutputs(), hiddenLayer.getErrors(), 8, Activation::relu);
    FullyConnectedLayer outputLayer(hiddenLayer2.getOutputs(), hiddenLayer2.getErrors(), 1, Activation::relu);
    double& networkOutput = outputLayer.getOutputs().front().get();
    double& networkError = outputLayer.getErrors().front().get();

    while(true) {
        std::cout<<"\n";

        // generates binary data and forward propogates it
        BinaryData binaryData;
        input = binaryData.binaryDigits;
        hiddenLayer.forwardPropogate();
        hiddenLayer2.forwardPropogate();
        outputLayer.forwardPropogate();
        
        // reports prediction
        for(const double& digit : binaryData.binaryDigits) {
            std::cout<<digit;
        }
        std::cout<<"\n"<<binaryData.baseTenSum<<"\n"<<networkOutput<<"\n";

        // trains model
        networkError = networkOutput - binaryData.baseTenSum;
        outputLayer.backPropogate(LEARNING_RATE);
        hiddenLayer2.backPropogate(LEARNING_RATE);
        hiddenLayer.backPropogate(LEARNING_RATE);
    }
}