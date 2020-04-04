#ifndef LAYER_HPP
#define LAYER_HPP

#include "Layer.hpp"
#include "../Neuron.hpp"
#include "../Activation.hpp"
#include <functional>
#include <vector>
#include <memory>

class Layer {
    public:
        Layer(std::vector<NeuronInterface>  backInterfaces, int size, const Activation::Activation& activationPair, bool batchNormalization);
        Layer(std::vector<NeuronInterface>  backInterfaces, int size, bool batchNormalization);
        virtual void forwardPropogate();
        virtual void backPropogate(const double& learningRate);
        std::vector<NeuronInterface> getInterfaces();

    protected:
        std::vector<std::shared_ptr<Neuron>> neurons;
        Layer();
        Layer(const Activation::Activation& activationPair, bool batchNormalization);

    private:
        const Activation::Activation& activation;
        void initializeNeurons(std::vector<NeuronInterface>  backInterfaces, int size);
        double scale;
        double shift;
        double scaleGrad;
        bool batchNorm;
};

#endif