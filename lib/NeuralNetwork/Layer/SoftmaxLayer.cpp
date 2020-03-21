#include "SoftmaxLayer.hpp"

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
        neurons[i]->softmax(numerators[i], denominator);
    }
}

void SoftmaxLayer::backPropogate(const double& learningRate) {
    for(auto& neuron : neurons) {
        neuron->backPropogate([](const double& x) -> double {return x * (double(1) - x);}, learningRate);
    }
}