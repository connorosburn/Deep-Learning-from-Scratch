#include "NeuralNetwork/Layer/Layer.hpp"
#include "NeuralNetwork/Layer/Layer.hpp"
#include "NeuralNetwork/Layer/SoftmaxLayer.hpp"
#include "NeuralNetwork/Loss.hpp"
#include <array>

const double LEARNING_RATE = 0.001;
const int EPOCHS = 2;
bool VERBOSE_READOUT = false;

std::vector<std::vector<double>> INPUT(28, std::vector<double>(28, 0));

Layer hiddenLayer(InputInterface(INPUT).interfaces, 100, Activation::relu);
SoftmaxLayer outputLayer(hiddenLayer.getInterfaces(), 10);

auto LOSS = Loss::binaryCrossEntropy;

std::vector<Layer*> LAYERS = {
    &hiddenLayer,
    &outputLayer
};