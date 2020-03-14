#include "FullyConnectedLayer.hpp"

FullyConnectedLayer::FullyConnectedLayer(std::vector<const double&> backOutputs, std::vector<const double&> backErrors, const Activation& activationPair):
activation(activationPair) {
    if(backOutputs.size() != backErrors.size()) {
        throw std::exception("Layer must receive the same number of output references as it receives error references");
    } else {
        initializeNeurons(backOutputs, backErrors);
    }
}

void FullyConnectedLayer::initializeNeurons(std::vector<const double&> outputs, std::vector<const double&> errors) {
    for(int i = 0; i < outputs.size(); i++) {
        neurons.emplace_back(outputs[i], errors[i]);
    }
}

void FullyConnectedLayer::forwardPropogate() {
    for(Neuron& neuron : neurons) {
        neuron.forwardPropogate(activation.activation);
    }
}
void backPropogate(const double& learningRate);