#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include <cmath>
/*
    NOTE: Activation derivatives currently take the 
    output that has already been run through the original 
    activation function, since all derivatives of activation
    functions CURRENTLY in use can be calculated from that value. 
    This avoids certain redundancies, but if nessecary, I'll go back
    and change it.
*/


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

    const Activation sigmoid (

        //sigmoid
        [](double x) -> double {
            return double(1) / (double(1) + std::exp(double(-1) * x));
        },
        
        //sigmoid derivative
        [](double x) -> double {
            return x * (double(1) - x);
        }
    );
}

#endif