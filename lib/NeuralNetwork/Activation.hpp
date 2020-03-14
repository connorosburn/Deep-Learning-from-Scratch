#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

namespace Activation {
    struct Activation {
        Activation(std::function<double(double)> act, std::function<double(double)> der):
        activation(act), derivative(der) {

        }
        std::function<double(double)> activation;
        std::function<double(double)> derivative;
    };

    const Activation relu (

        //relu
        [](double x) -> double {
            if(x > 0) {
                return x;
            } else {
                return 0;
            }
        },
        
        //relu derivative
        [](double x) -> double {
            if(x > 0) {
                return 1;
            } else {
                return 0;
            }
        }
    );
}

#endif