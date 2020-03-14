#include "Layer.hpp"
#include "Neuron.hpp"

class FullyConnectedLayer : public Layer {
    public:
        FullyConnectedLayer(std::vector<const double&> backOutputs, std::vector<double&> backErrors, const Activation& activationPair);
        void forwardPropogate();
        void backPropogate(const double& learningRate);
        std::vector<const double&> getOutputs();
        std::vector<double&> getErrors();

    private:
        const Activation& activation;
        std::vector<Neuron> neurons;
        void initializeNeurons(std::vector<const double&> outputs, std::vector<double&> errors);

};