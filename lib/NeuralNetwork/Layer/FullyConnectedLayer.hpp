#include "Layer.hpp"
#include "../Neuron.hpp"
#include "../Activation.hpp"

class FullyConnectedLayer : public Layer {
    public:
        FullyConnectedLayer(const std::vector<std::reference_wrapper<double>> backOutputs, std::vector<std::reference_wrapper<double>> backErrors, int size, const Activation::Activation& activationPair);
        void forwardPropogate();
        void backPropogate(const double& learningRate);
        const std::vector<std::reference_wrapper<double>> getOutputs();
        std::vector<std::reference_wrapper<double>> getErrors();

    private:
        const Activation::Activation& activation;
        std::vector<Neuron> neurons;
        void initializeNeurons(const std::vector<std::reference_wrapper<double>> outputs, std::vector<std::reference_wrapper<double>> errors, int size);

};