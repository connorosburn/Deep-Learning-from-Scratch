#include <iostream>
#include <vector>
#include <cmath>
#include "MNISTFashion/MNISTLoader.hpp"
#include "network_parameters.hpp"

int interpretNetworkOutput(const std::vector<NeuronInterface>& interfaces) {
    double highestOutput = 0;
    int highestIndex = 0;
    for(int i = 0; i < interfaces.size(); i++) {
        if(interfaces[i].output > highestOutput) {
            highestOutput = interfaces[i].output;
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

    auto networkInterface = LAYERS.back() -> getInterfaces();

    for(auto& data : loader.trainingData()) {
        for(int i = 0; i < INPUT.size(); i++) {
            INPUT[i] = data.image[i / 28][i % 28];
        }
        
        for(int i = 0; i < LAYERS.size(); i++) {
            LAYERS[i] -> forwardPropogate();
        }

        std::cout << "\n\nExpectation: " << data.label;
        std::cout << "\nPrediction: " << interpretNetworkOutput(networkInterface);
        std::cout << "\n\nRaw Outputs:";
        std::vector<double> expectation = labelVector(data.label);
        for(int i = 0; i < networkInterface.size(); i++) {
            std::cout << "\n" << networkInterface[i].output << " (" << expectation[i] << ")";
        }

        for(int i = 0; i < networkInterface.size(); i++) {
            networkInterface[i].errorAccumulator(LOSS.derivative(networkInterface[i].output, expectation[i]));
        }

        for(int i = LAYERS.size() - 1; i >= 0; i--) {
            LAYERS[i] -> backPropogate(LEARNING_RATE);
        }
    }
}