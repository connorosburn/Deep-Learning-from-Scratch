#include "NeuralNetwork/Layer/Layer.hpp"
#include "NeuralNetwork/Layer/Layer.hpp"
#include "NeuralNetwork/Loss.hpp"
#include <array>

const double LEARNING_RATE = 0.001;

std::array<double, 784> INPUT;
std::array<double, 784> TOTAL_ERROR;

Layer hiddenLayer({INPUT.begin(), INPUT.end()}, {TOTAL_ERROR.begin(), TOTAL_ERROR.end()}, 500, Activation::relu);
Layer outputLayer(hiddenLayer.getOutputs(), hiddenLayer.getErrors(), 10, Activation::sigmoid);

auto LOSS = Loss::binaryCrossEntropy;

std::vector<Layer*> LAYERS = {
    &hiddenLayer,
    &outputLayer
};