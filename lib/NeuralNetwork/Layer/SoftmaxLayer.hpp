#ifndef SOFTMAX_LAYER_HPP
#define SOFTMAX_LAYER_HPP

#include "Layer.hpp"

class SoftmaxLayer : public Layer {
    public:
        SoftmaxLayer(std::vector<NeuronInterface>  backInterfaces, int size): 
        Layer(backInterfaces, size, false) {};
        void forwardPropogate();
        void backPropogate(const double learningRate);
};

#endif