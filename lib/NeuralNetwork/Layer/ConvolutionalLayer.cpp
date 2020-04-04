#include "ConvolutionalLayer.hpp"
#include <iostream>

ConvolutionalLayer::ConvolutionalLayer(std::vector<std::vector<NeuronInterface>> interfaces, const int& filterWidth, const int& filterHeight, const int& nudgeDistance, const Activation::Activation& activationPair, bool batchNorm):
Layer(activationPair, batchNorm) {
    rows = 0;
    for(int y = 0; y < interfaces.size() - filterHeight; y += nudgeDistance) {
        for(int x = 0; x < interfaces[y].size() - filterWidth; x += nudgeDistance) {
            std::vector<NeuronInterface> neuronInterfaces;
            for(int i = 0; i < filterHeight; i++) {
                for(int j = 0; j < filterWidth; j++) {
                    neuronInterfaces.emplace_back(interfaces[i + y][j + x]);
                }
            }
            neurons.emplace_back(new Neuron(neuronInterfaces, !batchNorm));
        }
        rows++;
    }
}

std::vector<std::vector<NeuronInterface>> ConvolutionalLayer::getInterfaces2d() {
    std::vector<std::vector<NeuronInterface>> interfaces;
    for(int i = 0; i < rows; i++) {
        interfaces.emplace_back();
        for(int j = 0; j < (neurons.size() / rows); j++) {
            interfaces.back().emplace_back(neurons[(i * rows) + j]->getInterface());
        }
    }
    return interfaces;
}