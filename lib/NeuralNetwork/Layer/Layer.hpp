#ifndef FULLY_CONNECTED_LAYER_HPP
#define FULLY_CONNECTED_LAYER_HPP

#include "Layer.hpp"
#include "../Neuron.hpp"
#include "../Activation.hpp"
#include <functional>
#include <vector>

class Layer {
    public:
        Layer(std::vector<std::reference_wrapper<double>>  backOutputs, std::vector<std::reference_wrapper<double>> backErrors, int size, const Activation::Activation& activationPair);
        void forwardPropogate();
        void backPropogate(const double& learningRate);
        std::vector<std::reference_wrapper<double>>  getOutputs();
        std::vector<std::reference_wrapper<double>> getErrors();

    private:
        const Activation::Activation& activation;
        std::vector<Neuron> neurons;
        void initializeNeurons(std::vector<std::reference_wrapper<double>>  outputs, std::vector<std::reference_wrapper<double>> errors, int size);
};

#endif