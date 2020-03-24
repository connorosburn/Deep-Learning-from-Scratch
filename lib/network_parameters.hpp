#include "NeuralNetwork/Layer/Layer.hpp"
#include "NeuralNetwork/Layer/Layer.hpp"
#include "NeuralNetwork/Layer/SoftmaxLayer.hpp"
#include "NeuralNetwork/Loss.hpp"
#include <array>
#include "NeuralNetwork/Layer/ConvolutionalLayer.hpp"

const double LEARNING_RATE = 0.001;
const int EPOCHS = 10;
bool VERBOSE_READOUT = false;

std::vector<std::vector<double>> INPUT(28, std::vector<double>(28, 0));

ConvolutionalLayer c1(InputInterface(INPUT).interfaces, 5, 5, 1, Activation::relu);
ConvolutionalLayer c2(c1.getInterfaces2d(), 5, 5, 1, Activation::relu);
Layer f1(c2.getInterfaces(), 150, Activation::relu);
SoftmaxLayer ol(f1.getInterfaces(), 10);

auto LOSS = Loss::binaryCrossEntropy;

std::vector<Layer*> LAYERS = {
    &c1,
    &c2,
    &f1,
    &ol
};