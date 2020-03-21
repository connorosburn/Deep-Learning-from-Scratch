#include "Layer.hpp"

Layer::Layer(std::vector<std::reference_wrapper<double>>  backOutputs, std::vector<std::reference_wrapper<double>> backErrors, int size, const Activation::Activation& activationPair):
activation(activationPair) {
    initializeNeurons(backOutputs, backErrors, size);
}

Layer::Layer(std::vector<std::reference_wrapper<double>>  backOutputs, std::vector<std::reference_wrapper<double>> backErrors, int size):
activation(Activation::null) {
    initializeNeurons(backOutputs, backErrors, size);
}

void Layer::initializeNeurons(std::vector<std::reference_wrapper<double>>  outputs, std::vector<std::reference_wrapper<double>> errors, int size) {
    if(outputs.size() != errors.size()) {
        throw LayerError();
    } else {
        for(int i = 0; i < size; i++) {
            neurons.emplace_back(new Neuron(outputs, errors));
        }
    }
}

void Layer::forwardPropogate() {
    for(auto& neuron : neurons) {
        neuron->forwardPropogate(activation.activation);
    }
}

void Layer::backPropogate(const double& learningRate) {
    for(auto& neuron : neurons) {
        neuron->backPropogate(activation.derivative, learningRate);
    }
}

std::vector<std::reference_wrapper<double>>  Layer::getOutputs() {
    std::vector<std::reference_wrapper<double>> outputs;
    for(auto& neuron : neurons) {
        outputs.emplace_back(neuron->getOutput());
    }
    return outputs;
}

std::vector<std::reference_wrapper<double>> Layer::getErrors() {
    std::vector<std::reference_wrapper<double>> errors;
    for(auto& neuron : neurons) {
        errors.emplace_back(neuron->getError());
    }
    return errors;
}