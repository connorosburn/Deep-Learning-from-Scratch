#ifndef NEURON_HPP
#define NEURON_HPP

#include <vector>
#include "Weight.hpp"

class Neuron {
    public:
        Neuron(std::vector<NeuronInterface> backInterfaces, bool useBias);
        double productSum();
        void normalize(const double mean, const double variance, const double scale, const double shift);
        void forwardPropogate(std::function<double(double)> activation);
        void activate(std::function<double(double)> activation);
        void softmax(const double numerator, const double denominator);
        double backPropogate(std::function<double(double)> activationDerivative, const double learningRate);
        const NeuronInterface& getInterface();
        const double& getOutput() {return output;};
        
    private:
        std::vector<Weight> weights;
        double bias;
        double error;
        double output;
        bool adjustBias;
        void adjustWeights(const double delta, const double learningRate);
        NeuronInterface interface;

};

#endif