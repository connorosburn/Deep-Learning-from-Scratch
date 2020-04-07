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
    /*
        normalizes the value of neuron to a mean of 0 and standard deviation of 1
        given the mean and variance provided from outside.
        sqrt(variance + tinyConst) would just be the standard deviation, but that could 
        be 0 as an edge case, so it adds a tiny protection to prevent division by zero
    */
    output = (output - mean) / std::sqrt(variance + 0.0000001);
    /*
        re-adjusts the normalization based on factors decided at the layer level.
        These start out as 0 for shift and 1 for scale, effectively doing nothing, but
        these values get "trained" with the network to adjust the normalization level adaptively
    */
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
    /*
        error gets accumulated by layers in front during back propogation.
        After being used here, it is reset so that it can be summed next iteration
    */
    error = 0;
    return changeValue;
}

void Neuron::adjustWeights(const double delta, const double learningRate) {
    for(Weight& weight : weights) {
        weight.backInterface.errorAccumulator(weight.value * delta);
        weight.value -= (weight.backInterface.output * delta * learningRate);
    }
    //since the bias initializes at 0, if "adjustBias" is off, the neuron effectively doesnt have one.
    if(adjustBias) {
        bias -= delta * learningRate;
    }
}

const NeuronInterface& Neuron::getInterface() {
    return interface;
}