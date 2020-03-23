#include "Weight.hpp"
#include <random>


InputInterface::InputInterface(std::vector<std::reference_wrapper<double>> input) {
    for(std::reference_wrapper<double> ref : input) {
        interfaces.emplace_back([](const double& n){}, ref.get());
    }
}

Weight::Weight(NeuronInterface interface): backInterface(interface) {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<double> d{0,1};
    value = d(gen);
}

