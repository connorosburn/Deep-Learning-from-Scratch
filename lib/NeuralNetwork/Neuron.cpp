#include "Neuron.hpp"

Neuron::Neuron(std::vector<const double&> backOutputs, std::vector<const double&> backErrors): bias(0), error(0) {
    if(backOutputs.size() != backErrors.size()) {
        throw std::exception("Neuron must receive the same number of output references as it receives error references");
    } else {
        initializeWeights(backOutputs, backErrors);
    }
}

void Neuron::initializeWeights(std::vector<const double&> outputs, std::vector<const double&> errors) {
    for(int i = 0; i < backOutputs.size(); i++) {
        weights.emplace_back(outputs[i], errors[i]);
    }
}

double Neuron::productSum() {
    double sum = bias;
    for(const Weight& weight : weights) {
        sum += (weight.value * weight.backOutput);
    }
    return sum;
}

void Neuron::forwardPropogate(std::function<double(double)> activation) {
    output = activation(productSum());
}

void Neuron::backPropogate(std::function<double(double)> activationDerivative, const double& learningRate) {
    double delta = activationDerivative(output) * error;
    error = 0;
    adjustWeights(delta, learningRate);
}

void Neuron::adjustWeights(double delta, const double& learningRate) {
    for(Weight& weight : weights) {
        weight.backError += (weight.value * delta);
        weight.value -= (weight.backOutput * delta * learningRate);
    }
    bias -= delta * learningRate;
}