#ifndef LAYER_HPP
#define LAYER_HPP
#include <vector>

class Layer {
    public:
        Layer() {};
        virtual void forwardPropogate() = 0;
        virtual void backPropogate(const double& learningRate) = 0;
        virtual std::vector<std::reference_wrapper<double>>  getOutputs() = 0;
        virtual std::vector<std::reference_wrapper<double>> getErrors() = 0;

    protected:
        initializeNeurons(std::vector<std::reference_wrapper<double>>  outputs, std::vector<std::reference_wrapper<double>> errors, int size);
};

#endif