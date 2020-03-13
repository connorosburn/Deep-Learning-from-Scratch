#ifndef NEURON_HPP
#define NEURON_HPP

#include <vector>
#include "Weight.hpp"

class Neuron {
    public:
        Neuron(std::vector<const double&> backOutputs, std::vector<const double&> backErrors);
        double productSum();
        void forwardPropogate(std::function<double(double)> activation);
        void passToWeights(double delta, const double& learningRate);
        void backPropogate(std::function<double(double)> activationDerivative, const double& learningRate);
        const double& getOutput();
        double& getError();

    private:
        std::vector<Weight> weights;
        double bias;
        double output;
        double error;

};

#endif