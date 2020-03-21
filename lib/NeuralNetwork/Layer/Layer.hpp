#ifndef LAYER_HPP
#define LAYER_HPP

#include "Layer.hpp"
#include "../Neuron.hpp"
#include "../Activation.hpp"
#include <functional>
#include <vector>

struct LayerError : std::exception {
  const char* what() const noexcept {return "Neuron must receive the same number of output references as it receives error references\n";}
};

class Layer {
    public:
        Layer(std::vector<std::reference_wrapper<double>>  backOutputs, std::vector<std::reference_wrapper<double>> backErrors, int size, const Activation::Activation& activationPair);
        Layer(std::vector<std::reference_wrapper<double>>  backOutputs, std::vector<std::reference_wrapper<double>> backErrors, int size);
        virtual void forwardPropogate();
        virtual void backPropogate(const double& learningRate);
        virtual std::vector<std::reference_wrapper<double>>  getOutputs();
        virtual std::vector<std::reference_wrapper<double>> getErrors();

    protected:
        std::vector<Neuron> neurons;

    private:
        const Activation::Activation& activation;
        void initializeNeurons(std::vector<std::reference_wrapper<double>> outputs, std::vector<std::reference_wrapper<double>> errors, int size);
};

#endif