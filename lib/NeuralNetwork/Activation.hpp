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
        Activation(std::function<double(const double&)> act, std::function<double(const double&)> der):
        activation(act), derivative(der) {

        }
        std::function<double(const double&)> activation;
        std::function<double(const double&)> derivative;
    };

    const Activation null (
        [](const double& x) -> double {return 0;},
        [](const double& x) -> double {return 0;}
    );

    const Activation relu (

        //relu
        [](const double& x) -> double {
            if(x > 0) {
                return x;
            } else {
                return 0;
            }
        },
        
        //relu derivative
        [](const double& x) -> double {
            if(x > 0) {
                return 1;
            } else {
                return 0;
            }
        }
    );

    const Activation sigmoid (

        //sigmoid
        [](const double& x) -> double {
            const double OVERFLOW_MAX = 9;
            const double UNDERFLOW_MIN = -9;

            const double* stableX = &x;

            if(x > OVERFLOW_MAX) {
                stableX = &OVERFLOW_MAX;
            } else if(x < UNDERFLOW_MIN) {
                stableX = &UNDERFLOW_MIN;
            }

            return double(1) / (double(1) + std::exp(double(-1) * (*stableX)));
        },
        
        //sigmoid derivative
        [](const double& x) -> double {
            return x * (double(1) - x);
        }
    );
}

#endif