#include "Weight.hpp"
#include <random>

Weight::Weight(const double& output, double& error) backOutput(output), backError(error) {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<double> d{0,1};
    value = d(gen);
}