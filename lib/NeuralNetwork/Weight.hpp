#ifndef WEIGHT_HPP
#define WEIGHT_HPP

#include <functional>
#include <vector>

struct NeuronInterface {
    NeuronInterface(std::function<void(const double&)> errorAcc, const double& out):
    errorAccumulator(errorAcc), output(out) {};
    std::function<void(const double&)> errorAccumulator;
    const double& output;
};


struct InputInterface {
    InputInterface(std::vector<std::vector<double>>& input);
    std::vector<std::vector<NeuronInterface>> interfaces;
};

struct Weight {
    Weight(NeuronInterface interface);
    static double generateRandom();
    double value;
    NeuronInterface backInterface;
};

#endif