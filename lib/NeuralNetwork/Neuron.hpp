#ifndef NEURON_HPP
#define NEURON_HPP

#include <vector>
#include "Weight.hpp"

class Neuron {
    public:
        Neuron(std::vector<std::reference_wrapper<double>>  backOutputs, std::vector<std::reference_wrapper<double>> backErrors);
        void forwardPropogate(std::function<const double&(const double&)> activation);
        void backPropogate(std::function<const double&(const double&)> activationDerivative, const double& learningRate);
        double& getOutput();
        double& getError();

    protected:
        double productSum();

    private:
        std::vector<Weight> weights;
        double bias;
        double output;
        double error;
        void initializeWeights(std::vector<std::reference_wrapper<double>>  outputs, std::vector<std::reference_wrapper<double>> errors);
        void adjustWeights(double delta, const double& learningRate);

};

#endif