#include "Neuron.hpp"
#include <iostream>


struct NeuronError : std::exception {
  const char* what() const noexcept {return "Neuron must receive the same number of output references as it receives error references\n";}
};

Neuron::Neuron(std::vector<std::reference_wrapper<double>>  backOutputs, std::vector<std::reference_wrapper<double>> backErrors): bias(0), error(0) {
    if(backOutputs.size() != backErrors.size()) {
        throw NeuronError();
    } else {
        initializeWeights(backOutputs, backErrors);
    }
}

void Neuron::initializeWeights(std::vector<std::reference_wrapper<double>>  outputs, std::vector<std::reference_wrapper<double>> errors) {
    for(int i = 0; i < outputs.size(); i++) {
        weights.emplace_back(outputs[i].get(), errors[i].get());
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

double& Neuron::getOutput() {
    return output;
}

double& Neuron::getError() {
    return error;
}