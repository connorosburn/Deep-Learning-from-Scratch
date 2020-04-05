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
                return -1.0 * std::log(prediction);
            } else {
                return -1.0 * std::log(1.0 - prediction);
            }
        },
        
        // binary cross entropy derivative
        [](double prediction, double expectation) -> double {
            if(static_cast<int>(expectation) == 1) {
                return -1.0 * (1.0 / prediction);
            } else {
                return 1.0 / (1.0 - prediction);
            }
        }
    );

    const Loss crossEntropy (

        // cross entropy
        [](double prediction, double expectation) -> double {
            return -1.0 * expectation * std::log(prediction);
        },
        
        // cross entropy derivative
        [](double prediction, double expectation) -> double {
            return -1.0 * (expectation / prediction);
        }
    );
}

#endif