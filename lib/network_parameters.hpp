#include "NeuralNetwork/Layer/FullyConnectedLayer.hpp"
#include "NeuralNetwork/Layer/Layer.hpp"

const double LEARNING_RATE = 0.001;

std::array<double, 784> input;
std::array<double, 784> totalError;

FullyConnectedLayer hiddenLayer({input.begin(), input.end()}, {totalError.begin(), totalError.end()}, 500, Activation::relu);
FullyConnectedLayer outputLayer(hiddenLayer.getOutputs(), hiddenLayer.getErrors(), 10, Activation::sigmoid);

std::vector<Layer*> LAYERS = {
    &hiddenLayer,
    &outputLayer
};