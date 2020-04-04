#ifndef CONVOLUTIONAL_LAYER_HPP
#define CONVOLUTIONAL_LAYER_HPP

#include "Layer.hpp"

class ConvolutionalLayer : public Layer {
    public:
        ConvolutionalLayer(std::vector<std::vector<NeuronInterface>> interfaces, const int& filterWidth, const int& filterHeight, const int& nudgeDistance, const Activation::Activation& activationPair, bool batchNorm = true);
        std::vector<std::vector<NeuronInterface>> getInterfaces2d();
    
    private:
        int rows;
};

#endif