#ifndef LAYER_HPP
#define LAYER_HPP

class Layer {
    public:
        Layer() {};
        virtual void forwardPropogate() = 0;
        virtual void backPropogate(const double& learningRate) = 0;
};

#endif