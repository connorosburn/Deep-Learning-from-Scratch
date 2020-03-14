#include <iostream>
#include <vector>
#include <cmath>
#include <random>

std::vector<double> randomBinary(int length) {
    //totes copied/pasted (and modified), throwaway code anyway
    std::vector<double> output;
    static auto dev = std::random_device();
    static auto gen = std::mt19937{dev()};
    static auto dist = std::uniform_real_distribution<double>(0,1);
    for(int i = 0; i < length; i++) {
        if(dist(gen) < 0.5) {
            output.push_back(0);
        } else {
            output.push_back(1);
        }
    }
    return output;
}

double baseTenFromBinary(std::vector<double> binaryDigits) {
    double sum = 0;
    for(int i = 0; i < binaryDigits.size(); i++) {
        int position = binaryDigits.size() - 1 - i;
        sum += binaryDigits[position] * std::pow(2, i);
    }
    return sum;
}

double binarySum(std::vector<double> binaryDigits) {
    auto halfIter = binaryDigits.begin() + (binaryDigits.size() / 2);
    std::vector<double> firstBinary = std::vector<double>(binaryDigits.begin(), halfIter);
    std::vector<double> secondBinary = std::vector<double>(halfIter, binaryDigits.end());
    return baseTenFromBinary(firstBinary) + baseTenFromBinary(secondBinary);
}

struct BinaryData {
    BinaryData() {
        binaryDigits = randomBinary(8);
        baseTenSum = binarySum(binaryDigits);
    }
    std::vector<double> binaryDigits;
    double baseTenSum;
};

int main() {

}