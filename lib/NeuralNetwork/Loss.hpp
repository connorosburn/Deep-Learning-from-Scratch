#ifndef LOSS_HPP
#define LOSS_HPP

namespace Loss {
    struct Loss {
        Loss(std::function<double(double, double)> lossFunction, std::function<double(double, double)> lossDerivative): loss(lossFunction), derivative(lossDerivative) {};
        std::function<double(double, double)> loss;
        std::function<double(double, double)> derivative;
    };

    const Loss binaryCrossEntropy (

        // binary cross entropy
        [](double prediction, double expectation) -> double {
            if(expectation == 1) {
                return double(-1) * std::log(prediction);
            } else {
                return double(-1) * std::log(double(1) - prediction);
            }
        },
        
        // binary cross entropy derivative
        [](double prediction, double expectation) -> double {
            if(static_cast<int>(expectation) == 1) {
                return double(-1) * (double(1) / prediction);
            } else {
                return double(1) / (double(1) - prediction);
            }
        }
    );

    const Loss crossEntropy (

        // cross entropy
        [](double prediction, double expectation) -> double {
            return double(-1) * expectation * std::log(prediction);
        },
        
        // cross entropy derivative
        [](double prediction, double expectation) -> double {
            return double(-1) * (expectation / prediction);
        }
    );
}

#endif