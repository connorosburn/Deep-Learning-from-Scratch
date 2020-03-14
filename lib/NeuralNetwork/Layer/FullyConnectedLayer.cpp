#include "FullyConnectedLayer.hpp"

FullyConnectedLayer::FullyConnectedLayer(std::vector<const double&> backOutputs, std::vector<double&> backErrors, int size, const Activation& activationPair):
activation(activationPair) {
    if(backOutputs.size() != backErrors.size()) {
        throw std::exception("Layer must receive the same number of output references as it receives error references");
    } else {
        initializeNeurons(backOutputs, backErrors, size);
    }
}

void FullyConnectedLayer::initializeNeurons(std::vector<const double&> outputs, std::vector<double&> errors, int size) {
    for(int i = 0; i < size; i++) {
        neurons.emplace_back(outputs[i], errors[i]);
    }
}

void FullyConnectedLayer::forwardPropogate() {
    for(Neuron& neuron : neurons) {
        neuron.forwardPropogate(activation.activation);
    }
}
void FullyConnectedLayer::backPropogate(const double& learningRate) {
    for(Neuron& neuron : neurons) {
        neuron.backPropogate(activation.derivative, learningRate);
    }
}

std::vector<const double&> FullyConnectedLayer::getOutputs() {
    std::vector<const double&> outputs;
    for(const Neuron& neuron : neurons) {
        outputs.emplace_back(neuron.getOutput());
    }
    return outputs;
}

std::vector<double&> FullyConnectedLayer::getErrors() {
    std::vector<double&> errors;
    for(const Neuron& neuron : neurons) {
        errors.emplace_back(neuron.getError());
    }
    return errors;
}