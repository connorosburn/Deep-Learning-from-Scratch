#include "SoftmaxLayer.hpp"

void SoftmaxLayer::forwardPropogate() {
    double numerators[neurons.size()];

    double offset = 0;
    for(int i = 0; i < neurons.size(); i++) {
        numerators[i] = neurons[i].productSum();
        if(numerators[i] > offset) {
            offset = numerators[i];
        }
    }

    double denominator = 0;
    for(int i = 0; i < neurons.size(); i++) {
        numerators[i] = std::exp(numerators[i] - offset);
        denominator += numerators[i];
    }


    // this smells, but it works for now botht he under / overflow thing, and the getter reference garbage. ew.
    // DEFINITELY FIX!
    const double OVERFLOW_MAX = 0.99999;
    const double UNDERFLOW_MIN = 0.00001;

    for(int i = 0; i < neurons.size(); i++) {
        double output = numerators[i] / denominator;
        if(output > OVERFLOW_MAX) {
            neurons[i].getOutput() = OVERFLOW_MAX;
        } else if(output < UNDERFLOW_MIN) {
            neurons[i].getOutput() = UNDERFLOW_MIN;
        } else {
            neurons[i].getOutput() = output;
        }
    }
}

void SoftmaxLayer::backPropogate(const double& learningRate) {
    for(auto& neuron : neurons) {
        neuron.backPropogate([](const double& x) -> double {return x * (double(1) - x);}, learningRate);
    }
}