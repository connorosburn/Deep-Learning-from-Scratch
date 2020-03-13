#ifndef WEIGHT_HPP
#define WEIGHT_HPP

struct Weight {
    Weight(const double& output, double& error);
    double value;
    const double& backOutput;
    double& backError;
};

#endif