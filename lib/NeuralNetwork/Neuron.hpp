#ifndef NEURON_HPP
#define NEURON_HPP

#include <vector>
#include "Weight.hpp"

class Neuron {
    public:
        Neuron(std::vector<std::reference_wrapper<double>>  backOutputs, std::vector<std::reference_wrapper<double>> backErrors);
        double productSum();
        void forwardPropogate(std::function<const double&(const double&)> activation);
        void softmax(const double& numerator, const double& denominator);
        void backPropogate(std::function<const double&(const double&)> activationDerivative, const double& learningRate);
        double& getOutput();
        double& getError();
        
    private:
        std::vector<Weight> weights;
        double bias;
        double error;
        double output;
        void initializeWeights(std::vector<std::reference_wrapper<double>>  outputs, std::vector<std::reference_wrapper<double>> errors);
        void adjustWeights(double delta, const double& learningRate);

};

#endif