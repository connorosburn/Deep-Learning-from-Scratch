#ifndef SOFTMAX_LAYER_HPP
#define SOFTMAX_LAYER_HPP

#include "Layer.hpp"

class SoftmaxLayer : public Layer {
    public:
        SoftmaxLayer(std::vector<std::reference_wrapper<double>>  backOutputs, std::vector<std::reference_wrapper<double>> backErrors, int size);
        void forwardPropogate();

    private:
        class SoftmaxNeuron : public Neuron {
            public:
                const double& calculateNumerator();
                void softmax(const double& denominator);

            private:
                double numerator;
        };

        std::vector<SoftmaxNeuron> neurons;
};

#endif