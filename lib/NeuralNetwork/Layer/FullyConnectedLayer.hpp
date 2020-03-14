#include "Layer.hpp"

class FullyConnectedLayer : public Layer {
    public:
        FullyConnectedLayer(std::vector<const double&> backOutputs, std::vector<const double&> backErrors, const Activation& activationPair);
        void forwardPropogate();
        void backPropogate(const double& learningRate);

    private:
        const Activation& activation;

};