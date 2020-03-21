#ifndef SOFTMAX_LAYER_HPP
#define SOFTMAX_LAYER_HPP

#include "Layer.hpp"

class SoftmaxLayer : public Layer {
    public:
        SoftmaxLayer(std::vector<std::reference_wrapper<double>>  backOutputs, std::vector<std::reference_wrapper<double>> backErrors, int size): 
        Layer(backOutputs, backErrors, size) {};
        void forwardPropogate();
        void backPropogate(const double& learningRate);
};

#endif