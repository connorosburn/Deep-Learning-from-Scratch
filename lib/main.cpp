#include <iostream>
#include "Layer/FullyConnectedLayer.hpp"

std::vector<double> randomBinary() {

}

double binarySum(std::vector<double> binaryDigits) {
    
}

struct BinaryData {
    BinaryData() {
        binaryDigits = randomBinary(10);
        baseTenSum = binarySum(binaryDigits);
    }
    std::vector<double> binaryDigits;
    double baseTenSum;
};

int main() {
    // machine learns here
}