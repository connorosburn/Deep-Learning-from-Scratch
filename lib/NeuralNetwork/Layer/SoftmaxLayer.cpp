#include "SoftmaxLayer.hpp"

double softmax(const double numerator, const double denominator) {
    const double MAX = 0.9999999;
    const double MIN = 0.0000001;
    double output = numerator / denominator;
    std::clamp(output, MIN, MAX);
    return output;
}

void SoftmaxLayer::forwardPropogate() {
    double numerators[neurons.size()];

    double offset = 0;
    for(int i = 0; i < neurons.size(); i++) {
        numerators[i] = neurons[i]->productSum();
        if(numerators[i] > offset) {
            offset = numerators[i];
        }
    }

    double denominator = 0;
    for(int i = 0; i < neurons.size(); i++) {
        numerators[i] = std::exp(numerators[i] - offset);
        denominator += numerators[i];
    }

    for(int i = 0; i < neurons.size(); i++) {
        neurons[i]->activate([&numerators, &i, &denominator](double z) {
            const double MAX = 0.9999999;
            const double MIN = 0.0000001;
            double output = numerators[i] / denominator;
            std::clamp(output, MIN, MAX);
            return output;
        });
    }
}

void SoftmaxLayer::backPropogate(const double learningRate) {
    for(auto& neuron : neurons) {
        neuron->backPropogate([](const double x) -> double {return x * (1.0 - x);}, learningRate);
    }
}