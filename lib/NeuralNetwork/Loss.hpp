#ifndef LOSS_HPP
#define LOSS_HPP

namespace Loss {
    struct Loss {
        Loss(std::function<double(const double&, const double&)> lossFunction, std::function<double(const double&, const double&)> lossDerivative): loss(lossFunction), derivative(lossDerivative) {};
        std::function<double(const double&, const double&)> loss;
        std::function<double(const double&, const double&)> derivative;
    };

    const Loss binaryCrossEntropy (

        // binary cross entropy
        [](const double& prediction, const double& expectation) -> double {
            if(expectation == 1) {
                return double(-1) * std::log(prediction);
            } else {
                return double(-1) * std::log(double(1) - prediction);
            }
        },
        
        // binary cross entropy derivative
        [](const double& prediction, const double& expectation) -> double {
            if(expectation == 1) {
                return double(-1) * (double(1) / prediction);
            } else {
                return double(1) / (double(1) - prediction);
            }
        }
    );

    const Loss crossEntropy (

        // cross entropy
        [](const double& prediction, const double& expectation) -> double {
            return double(-1) * expectation * std::log(prediction);
        },
        
        // cross entropy derivative
        [](const double& prediction, const double& expectation) -> double {
            return double(-1) * (expectation / prediction);
        }
    );
}

#endif