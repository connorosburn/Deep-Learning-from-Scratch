#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include "NeuralNetwork/Layer/FullyConnectedLayer.hpp"
#include "NeuralNetwork/Activation.hpp"
#include <functional>
#include "MNISTFashion/MNISTLoader.hpp"
#include "network_parameters.hpp"

void binaryCrossEntropyDerivative(const std::vector<double>& prediction, const std::vector<double>& expectation, std::vector<std::reference_wrapper<double>> errorRef) {
    for(int i = 0; i < prediction.size(); i++) {
        if(expectation[i] > 0) {
            errorRef[i].get() = double(-1) * (double(1) / prediction[i]);
        } else {
            errorRef[i].get() = double(1) / (double(1) - prediction[i]);
        }
    }
}

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
        for(int i = 0; i < input.size(); i++) {
            input[i] = data.image[i / 28][i % 28];
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

        binaryCrossEntropyDerivative(std::vector<double>(networkOutput.begin(), networkOutput.end()), expectation, networkError);
        outputLayer.backPropogate(LEARNING_RATE);
        hiddenLayer.backPropogate(LEARNING_RATE);

        for(int i = LAYERS.size() - 1; i >= 0; i--) {
            LAYERS[i] -> backPropogate(LEARNING_RATE);
        }
    }
}