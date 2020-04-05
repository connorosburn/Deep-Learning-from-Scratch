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

void verboseReadout(const std::string& setName, const int label, const std::vector<NeuronInterface>& networkInterface, const std::vector<double>& expectation, const float percentageComplete) {
    std::cout << "\n\nExpectation: " << label;
    std::cout << "\nPrediction: " << interpretNetworkOutput(networkInterface);
    std::cout << "\n\nRaw Outputs:";
    for(int i = 0; i < networkInterface.size(); i++) {
        std::cout << "\n" << networkInterface[i].output << " (" << expectation[i] << ")";
    }
    std::cout<<"\n"<<setName <<" "<<percentageComplete<<" percent complete";
}

float percentThrough(float n, float c) {
    return (n / c) * 100.0;
}

void smallReadout(const std::string& cycleName, const int& exampleNumber, const int& exampleCount) {
    if(exampleNumber == 0) {
        std::cout<<"\n"<<cycleName<<" [";
    } else if(static_cast<int>(percentThrough(exampleNumber, exampleCount)) > static_cast<int>(percentThrough(exampleNumber - 1, exampleCount))) {
        std::cout<<"|";
        std::cout.flush();
    }
    if(exampleNumber == exampleCount - 1) {
        std::cout<<"]\n";
    }
}
void readout(const std::string& setName, std::vector<NeuronInterface>& networkInterface, const std::vector<double>& expectation, const std::vector<DataPair>& data, const int dataIndex) {
    if(VERBOSE_READOUT) {
        verboseReadout(setName, data[dataIndex].label, networkInterface, expectation, percentThrough(dataIndex, data.size()));
    } else {
        smallReadout(setName, dataIndex, data.size());
    }
}

bool predictExample(std::vector<NeuronInterface>& networkInterface, int label) {        
    for(Layer* layer : LAYERS) {
        layer -> forwardPropogate();
    }
    return interpretNetworkOutput(networkInterface) == label;
}

void backwardPass(std::vector<NeuronInterface>& networkInterface, const std::vector<double>& expectation) {
    for(int i = 0; i < networkInterface.size(); i++) {
        // "accumulator" only refers to how error gets accumulated from multiple sources inside the network.
        // It is reset each pass, so it is functionally a setter here.
        networkInterface[i].errorAccumulator(LOSS.derivative(networkInterface[i].output, expectation[i]));
    }

    for(int i = LAYERS.size() - 1; i >= 0; i--) {
        LAYERS[i] -> backPropogate(LEARNING_RATE);
    }
}

void predictDataset(std::string setName, std::vector<DataPair> data, bool training) {
    auto networkInterface = LAYERS.back() -> getInterfaces();
    int numberCorrect = 0;
    for(int i = 0; i < data.size(); i++) {
        INPUT.assign(data[i].image.begin(), data[i].image.end());
        if(predictExample(networkInterface, data[i].label)) {
            numberCorrect++;
        }
        std::vector<double> expectation = labelVector(data[i].label);
        readout(setName, networkInterface, expectation, data, i);
        if(training) {
            backwardPass(networkInterface, expectation);
        }
    }
    std::cout<<"\nAccuracy: "<<percentThrough(numberCorrect, data.size())<<" percent\n";
}

int main() {
    MNISTLoader loader;
    for(int i = 0; i < EPOCHS; i++) {
        predictDataset("Epoch " + std::to_string(i + 1), loader.trainingData(), true);
    }
    predictDataset("Test", loader.testData(), false);
}