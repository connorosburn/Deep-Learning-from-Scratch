#include "Layer.hpp"

Layer::Layer(std::vector<NeuronInterface>  backInterfaces, int size, const Activation::Activation& activationPair):
activation(activationPair) {
    initializeNeurons(backInterfaces, size);
}

Layer::Layer(std::vector<NeuronInterface>  backInterfaces, int size):
activation(Activation::null) {
    initializeNeurons(backInterfaces, size);
}

void Layer::initializeNeurons(std::vector<NeuronInterface>  backInterfaces, int size) {
    for(int i = 0; i < size; i++) {
        neurons.emplace_back(new Neuron(backInterfaces));
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

std::vector<NeuronInterface>  Layer::getInterfaces() {
    std::vector<NeuronInterface> outputs;
    for(auto& neuron : neurons) {
        outputs.emplace_back(neuron->getInterface());
    }
    return outputs;
}