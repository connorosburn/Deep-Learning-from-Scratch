#include "Layer.hpp"
#include <iostream>

Layer::Layer(std::vector<NeuronInterface>  backInterfaces, int size, const Activation::Activation& activationPair, bool batchNormalization):
activation(activationPair), scale(1), shift(0), batchNorm(batchNormalization) {
    initializeNeurons(backInterfaces, size);
}

Layer::Layer(std::vector<NeuronInterface>  backInterfaces, int size, bool batchNormalization):
activation(Activation::null), scale(1), shift(0), batchNorm(batchNormalization) {
    initializeNeurons(backInterfaces, size);
}

Layer::Layer(): activation(Activation::null), scale(1), shift(0) {

}

Layer::Layer(const Activation::Activation& activationPair, bool batchNormalization): activation(activationPair), scale(1), shift(0), batchNorm(batchNormalization) {

}

void Layer::initializeNeurons(std::vector<NeuronInterface>  backInterfaces, int size) {
    for(int i = 0; i < size; i++) {
        //only uses a bias in the neuron if batch normalization is off
        neurons.emplace_back(new Neuron(backInterfaces, !batchNorm));
    }
}

void Layer::forwardPropogate() {
    if(batchNorm) {
        /*
            batch normalization works on the whole layer even for layers that aren't fully connected.
            since convolutional layers inherit this function, it works for both
        */
        double mean = 0;
        for(auto& neuron : neurons) {
            mean += neuron->productSum();
        }
        mean /= static_cast<double>(neurons.size());

        double variance = 0;
        scaleGrad = 0;
        for(auto& neuron : neurons) {
            double seed = neuron->getOutput() - mean;
            variance += std::pow(seed, 2.0);
            scaleGrad += 2.0 * seed;
        }
        variance /= static_cast<double>(neurons.size());
        scaleGrad /= static_cast<double>(neurons.size());
        for(auto& neuron : neurons) {
            neuron->normalize(mean, variance, scale, shift);
            neuron->activate(activation.activation);
        }
    } else {
        for(auto& neuron : neurons) {
            neuron->forwardPropogate(activation.activation);
        }
    }
}

void Layer::backPropogate(const double learningRate) {
    /*
        error has to be calculated for the whole layer, even on non-fully connected layers
        to adjust the scale and shift factors appropriately
    */
    double layerError = 0;
    for(auto& neuron : neurons) {
        layerError += neuron->backPropogate(activation.derivative, learningRate);
    }
    if(batchNorm) {
        // adjusts the scale and shift factors by the error times their gradient (which is just 1 for shift)
        // tones down the error by a factor of the number of neurons since it accumulates from all of them
        const double adjustment = layerError * learningRate * std::pow(double(neurons.size()), double(-1));
        scale -= adjustment * scaleGrad;
        shift -= adjustment;
    }
}

std::vector<NeuronInterface>  Layer::getInterfaces() {
    std::vector<NeuronInterface> outputs;
    for(auto& neuron : neurons) {
        outputs.emplace_back(neuron->getInterface());
    }
    return outputs;
}