#include "Weight.hpp"
#include <random>


InputInterface::InputInterface(std::vector<std::vector<double>>& input) {
    for(std::vector<double>& row : input) {
        interfaces.emplace_back();
        for(double& pixel : row) {
            interfaces.back().emplace_back([](const double& n){}, pixel);
        }
    }
}

Weight::Weight(NeuronInterface interface): backInterface(interface) {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<double> d{0,1};
    value = d(gen);
}

