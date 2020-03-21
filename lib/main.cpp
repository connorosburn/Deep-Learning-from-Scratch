#include <iostream>
#include <vector>
#include <cmath>
#include "MNISTFashion/MNISTLoader.hpp"
#include "network_parameters.hpp"

int interpretNetworkOutput(const std::vector<double>& networkOutput) {
    double highestOutput = 0;
    int highestIndex = 0;
    for(int i = 0; i < networkOutput.size(); i++) {
        if(networkOutput[i] > highestOutput) {
            highestOutput = networkOutput[i];
            highestIndex = i;
        }
    }
    return highestIndex;
}

std::vector<double> labelVector(int label) {
    std::vector<double> output = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    output[label] = 1;
    return output;
}

int main() {
    MNISTLoader loader;

    auto networkOutput = LAYERS.back() -> getOutputs();
    auto networkError = LAYERS.back() -> getErrors();

    for(auto& data : loader.trainingData()) {
        for(int i = 0; i < INPUT.size(); i++) {
            INPUT[i] = data.image[i / 28][i % 28];
        }
        
        for(int i = 0; i < LAYERS.size(); i++) {
            LAYERS[i] -> forwardPropogate();
        }

        std::cout << "\n\nExpectation: " << data.label;
        std::cout << "\nPrediction: " << interpretNetworkOutput(std::vector<double>(networkOutput.begin(), networkOutput.end()));
        std::cout << "\n\nRaw Outputs:";
        std::vector<double> expectation = labelVector(data.label);
        for(int i = 0; i < networkOutput.size(); i++) {
            std::cout << "\n" << networkOutput[i] << " (" << expectation[i] << ")";
        }

        for(int i = 0; i < networkOutput.size(); i++) {
            networkError[i].get() = LOSS.derivative(networkOutput[i], expectation[i]);
        }

        for(int i = LAYERS.size() - 1; i >= 0; i--) {
            LAYERS[i] -> backPropogate(LEARNING_RATE);
        }
    }
}