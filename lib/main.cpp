#include <iostream>
#include <algorithm>
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

void verboseReadout(const std::string& setName, const int& label, const std::vector<NeuronInterface>& networkInterface, const std::vector<double>& expectation, const float& percentageComplete) {
    std::cout << "\n\nExpectation: " << label;
    std::cout << "\nPrediction: " << interpretNetworkOutput(networkInterface);
    std::cout << "\n\nRaw Outputs:";
    for(int i = 0; i < networkInterface.size(); i++) {
        std::cout << "\n" << networkInterface[i].output << " (" << expectation[i] << ")";
    }
    std::cout<<"\n"<<setName <<" "<<percentageComplete<<" percent complete";
}

int percentThrough(float n, float c) {
    return (n / c) * float(100);
}

void smallReadout(const std::string& cycleName, const int& exampleNumber, const int& exampleCount) {
    if(exampleNumber == 0) {
        std::cout<<"\n"<<cycleName<<" [";
    } else if(percentThrough(exampleNumber, exampleCount) > percentThrough(exampleNumber - 1, exampleCount)) {
        std::cout<<"|";
        std::cout.flush();
    }
    if(exampleNumber == exampleCount - 1) {
        std::cout<<"]\n";
    }
}

void predictDataset(std::string setName, std::vector<DataPair> data, bool training) {
    auto networkInterface = LAYERS.back() -> getInterfaces();
    int numberCorrect = 0;
    for(int d = 0; d < data.size(); d++) {
        INPUT.assign(data[d].image.begin(), data[d].image.end());
        
        for(int i = 0; i < LAYERS.size(); i++) {
            LAYERS[i] -> forwardPropogate();
        }

        std::vector<double> expectation = labelVector(data[d].label);

        if(VERBOSE_READOUT) {
            float percentageComplete = (float(d) / float(data.size())) * float(100);
            verboseReadout(setName, data[d].label, networkInterface, expectation, percentageComplete);
        } else {
            smallReadout(setName, d, data.size());
        }

        if(interpretNetworkOutput(networkInterface) == data[d].label) {
            numberCorrect++;
        }
        if(training) {
            for(int i = 0; i < networkInterface.size(); i++) {
                networkInterface[i].errorAccumulator(LOSS.derivative(networkInterface[i].output, expectation[i]));
            }

            for(int i = LAYERS.size() - 1; i >= 0; i--) {
                LAYERS[i] -> backPropogate(LEARNING_RATE);
            }
        }
    }
    std::cout<<"\nAccuracy: "<<(float(numberCorrect) / float(data.size())) * float(100)<<" percent\n";
}

int main() {
    MNISTLoader loader;

    auto data = loader.trainingData();
    auto networkInterface = LAYERS.back() -> getInterfaces();
    for(int i = 0; i < EPOCHS; i++) {
        predictDataset("Epoch " + std::to_string(i + 1), loader.trainingData(), true);
    }

    predictDataset("Test", loader.testData(), false);
}