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

void binaryCrossEntropyDerivative(const std::vector<double>& prediction, const std::vector<double>& expectation, std::vector<std::reference_wrapper<double>> errorRef) {
    for(int i = 0; i < prediction.size(); i++) {
        if(expectation[i] == 1) {
            errorRef[i] = double(1) / prediction[i];
        } else {
            errorRef[i] = double(1) / (double(1) - prediction[i]);
        }
    }
}

int main() {
    const double LEARNING_RATE = 0.000001;

    std::vector<double> input(8, 0);
    std::vector<double> totalError(8, 0);
    FullyConnectedLayer hiddenLayer({input.begin(), input.end()}, {totalError.begin(), totalError.end()}, 8, Activation::relu);
    FullyConnectedLayer outputLayer(hiddenLayer.getOutputs(), hiddenLayer.getErrors(), 5, Activation::sigmoid);
    auto& networkOutput = outputLayer.getOutputs();
    auto& networkError = outputLayer.getErrors();

    while(true) {
        // generates binary data and forward propogates it
        BinaryData binaryData;
        input = binaryData.binaryDigits;
        hiddenLayer.forwardPropogate();
        outputLayer.forwardPropogate();
        
        // reports prediction
        std::vector<double> prediction = interpretNetworkOutput(networkOutput);
        std::cout<<"\n";
        std::cout<<"Input: ";
        for(const double& digit : binaryData.binaryDigits) {
            std::cout<<digit;
        }
        std::cout<<"\nExpectation: ";
        for(const double& digit : binaryData.binarySum) {
            std::cout<<digit;
        }
        std::cout<<"\nPrediction: ";
        for(const double& digit : prediction) {
            std::cout<<digit;
        }
        std::cout<<"\nRaw Outputs:"
        for(int i = 0; i < networkOutput.size(); i++) {
            std::cout<<"\n"<<networkOutput[i]<<"("<<prediction[i]<<")";
        }
        std::cout<<"\n"

        // trains model
        binaryCrossEntropyDerivative(networkOutput, binaryData.binarySum, networkError);
        outputLayer.backPropogate(LEARNING_RATE);
        hiddenLayer.backPropogate(LEARNING_RATE);
    }
}