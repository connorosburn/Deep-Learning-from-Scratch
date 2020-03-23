#include "Neuron.hpp"

Neuron::Neuron(std::vector<NeuronInterface>  backInterfaces): 
bias(0), 
error(0),
interface([this](const double& addition) -> void {this->error += addition;}, output) {
    for(const auto& interface : backInterfaces) {
        weights.emplace_back(interface);
    }
}

double Neuron::productSum() {
    double sum = bias;
    for(const Weight& weight : weights) {
        sum += (weight.value * weight.backInterface.output);
    }
    return sum;
}

void Neuron::forwardPropogate(std::function<const double&(const double&)> activation) {
    output = activation(productSum());
}

void Neuron::softmax(const double& numerator, const double& denominator) {
    // this smells. fix, but figure out a way to make the rounding to 0 be a problem
    const double OVERFLOW_MAX = 0.9999999;
    const double UNDERFLOW_MIN = 0.0000001;

    double raw = numerator / denominator;
    if(raw > OVERFLOW_MAX) {
        output = OVERFLOW_MAX;
    } else if(raw < UNDERFLOW_MIN) {
        output = UNDERFLOW_MIN;
    } else {
        output = raw;
    }
}

void Neuron::backPropogate(std::function<const double&(const double&)> activationDerivative, const double& learningRate) {
    adjustWeights(activationDerivative(output) * error, learningRate);
    error = 0;
}

void Neuron::adjustWeights(double delta, const double& learningRate) {
    for(Weight& weight : weights) {
        weight.backInterface.errorAccumulator(weight.value * delta);
        weight.value -= (weight.backInterface.output * delta * learningRate);
    }
    bias -= delta * learningRate;
}

const NeuronInterface& Neuron::getInterface() {
    return interface;
}