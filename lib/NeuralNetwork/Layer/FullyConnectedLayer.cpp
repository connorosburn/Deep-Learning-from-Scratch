#include "FullyConnectedLayer.hpp"

FullyConnectedLayer::FullyConnectedLayer(const std::vector<std::reference_wrapper<double>> backOutputs, std::vector<std::reference_wrapper<double>> backErrors, int size, const Activation::Activation& activationPair):
activation(activationPair) {
    if(backOutputs.size() != backErrors.size()) {
        throw std::exception("Layer must receive the same number of output references as it receives error references");
    } else {
        initializeNeurons(backOutputs, backErrors, size);
    }
}

void FullyConnectedLayer::initializeNeurons(const std::vector<std::reference_wrapper<double>> outputs, std::vector<std::reference_wrapper<double>> errors, int size) {
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

const std::vector<std::reference_wrapper<double>> FullyConnectedLayer::getOutputs() {
    std::vector<const double&> outputs;
    for(const Neuron& neuron : neurons) {
        outputs.emplace_back(neuron.getOutput());
    }
    return outputs;
}

std::vector<std::reference_wrapper<double>> FullyConnectedLayer::getErrors() {
    std::vector<double&> errors;
    for(const Neuron& neuron : neurons) {
        errors.emplace_back(neuron.getError());
    }
    return errors;
}