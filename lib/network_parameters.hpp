#include "NeuralNetwork/Layer/Layer.hpp"
#include "NeuralNetwork/Layer/Layer.hpp"
#include "NeuralNetwork/Layer/SoftmaxLayer.hpp"
#include "NeuralNetwork/Loss.hpp"
#include <array>

const double LEARNING_RATE = 0.001;

std::array<double, 784> INPUT;
std::array<double, 784> TOTAL_ERROR;


//yea, when I pull this out to the input layer it feels weird
InputInterface INPUT_INTERFACE({INPUT.begin(), INPUT.end()});

Layer hiddenLayer(INPUT_INTERFACE.interfaces, 500, Activation::relu);
SoftmaxLayer outputLayer(hiddenLayer.getInterfaces(), 10);

auto LOSS = Loss::binaryCrossEntropy;

std::vector<Layer*> LAYERS = {
    &hiddenLayer,
    &outputLayer
};