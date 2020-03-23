#ifndef NEURON_HPP
#define NEURON_HPP

#include <vector>
#include "Weight.hpp"

class Neuron {
    public:
        Neuron(std::vector<NeuronInterface> backInterfaces);
        double productSum();
        void forwardPropogate(std::function<const double&(const double&)> activation);
        void softmax(const double& numerator, const double& denominator);
        void backPropogate(std::function<const double&(const double&)> activationDerivative, const double& learningRate);
        const NeuronInterface& getInterface();
        
    private:
        std::vector<Weight> weights;
        double bias;
        double error;
        double output;
        void adjustWeights(double delta, const double& learningRate);
        NeuronInterface interface;

};

#endif