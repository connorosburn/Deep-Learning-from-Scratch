#ifndef LAYER_HPP
#define LAYER_HPP

class Layer {
    public:
        virtual void forwardPropogate();
        virtual void backPropogate(const double& learningRate);
};

#endif