#include "NeuralNetwork/Layer/Layer.hpp"
#include "NeuralNetwork/Layer/Layer.hpp"
#include "NeuralNetwork/Layer/SoftmaxLayer.hpp"
#include "NeuralNetwork/Loss.hpp"
#include <array>
#include "NeuralNetwork/Layer/ConvolutionalLayer.hpp"
#include "NeuralNetwork/Layer/MaxPoolingLayer.hpp"

const double LEARNING_RATE = 0.001;
/*
    how many training examples it looks at before running the test set.
    training examples are selected randomly
*/
const int TRAINING_EXAMPLES = 10000;
/*
    how many times it repeats the number of training examples above, basically allowing you to check
    how you are doing periodically
*/
const int REPEAT_FACTOR = 60;

//get ready to have a flooded terminal if you set this to true, shows breakdown for every example
const bool VERBOSE_READOUT = false;

std::vector<std::vector<double>> INPUT(28, std::vector<double>(28, 0));

/*
    any type of layer that could have a 2d output shape gets the output interfaces via
    getInterfaces2d, wheres getInterfaces always outputs a flat vector

    NOTE: REMEMBER TO MAKE SURE LAYERS ARE INCLUDED IN THE VECTOR OF LAYER POINTERS-- 
    ALSO MAKE THAT LESS ANNOYING AT SOME POINT
*/
ConvolutionalLayer c1(InputInterface(INPUT).interfaces, 3, 3, 1, Activation::relu, true);
ConvolutionalLayer c2(c1.getInterfaces2d(), 3, 3, 1, Activation::relu, true);
MaxPoolingLayer p1(c2.getInterfaces2d(), 2, 2);
Layer f1(p1.getInterfaces(), 121, Activation::relu, true);
SoftmaxLayer ol(f1.getInterfaces(), 10);

auto LOSS = Loss::binaryCrossEntropy;

std::vector<Layer*> LAYERS = {
    &c1,
    &c2,
    &p1,
    &f1,
    &ol
};