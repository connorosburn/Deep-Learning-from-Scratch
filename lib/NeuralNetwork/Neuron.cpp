#include "Neuron.hpp"
#include <cmath>
#include <cfloat>
#include <iostream>

Neuron::Neuron(std::vector<NeuronInterface>  backInterfaces, bool useBias): 
error(0),
bias(0),
adjustBias(useBias),
interface([this](double addition) -> void {this->error += addition;}, output) {
    for(const auto& interface : backInterfaces) {
        weights.emplace_back(interface);
    }
}

double Neuron::productSum() {
    output = bias;
    for(const Weight& weight : weights) {
        output += (weight.value * weight.backInterface.output);
    }
    return output;
}

void Neuron::normalize(const double mean, const double variance, const double scale, const double shift) {
    output = (output - mean) / std::sqrt(variance + 0.0000001);
    output = (output * scale) + shift;
}

void Neuron::forwardPropogate(std::function<double(double)> activation) {
    output = activation(productSum());
}

void Neuron::activate(std::function<double(double)> activation) {
    output = activation(output);
}

double Neuron::backPropogate(std::function<double(double)> activationDerivative, const double learningRate) {
    const double changeValue = activationDerivative(output) * error;
    adjustWeights(changeValue, learningRate);
    error = 0;
    return changeValue;
}

void Neuron::adjustWeights(const double delta, const double learningRate) {
    for(Weight& weight : weights) {
        weight.backInterface.errorAccumulator(weight.value * delta);
        weight.value -= (weight.backInterface.output * delta * learningRate);
    }
    if(adjustBias) {
        bias -= delta * learningRate;
    }
}

const NeuronInterface& Neuron::getInterface() {
    return interface;
}