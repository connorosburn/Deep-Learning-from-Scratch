#ifndef MAX_POOLING_LAYER_HPP
#define MAX_POOLING_LAYER_HPP

#include "Layer.hpp"

class MaxPoolingLayer : public Layer {
    public:
        MaxPoolingLayer(std::vector<std::vector<NeuronInterface>> interfaces, const int poolWidth, const int poolHeight);
        void forwardPropogate();
        void backPropogate(const double learningRate);
        std::vector<NeuronInterface> getInterfaces();
        std::vector<std::vector<NeuronInterface>> getInterfaces2d();

    private:
        class PoolingCluster {
            public:
                PoolingCluster(std::vector<NeuronInterface> pool);
                NeuronInterface getInterface();
                void forwardPropogate();
                void backPropogate();

            private:
                double output;
                double error;
                std::vector<NeuronInterface> backInterfaces;
                NeuronInterface *lastMax;
        };
        std::vector<PoolingCluster> clusters;
        int rows;
};

#endif