#include "MaxPoolingLayer.hpp"
#include <iostream>

MaxPoolingLayer::MaxPoolingLayer(std::vector<std::vector<NeuronInterface>> interfaces, const int poolWidth, const int poolHeight) {
    rows = 0;
    for(int i = 0; i < interfaces.size(); i += poolHeight) {
        rows++;
        for(int j = 0; j < interfaces[i].size(); j += poolWidth) {
            std::vector<NeuronInterface> pool;
            for(int y = 0; y < poolHeight && i + y < interfaces.size(); y++) {
                for(int x = 0; x < poolWidth && j + x < interfaces[i + y].size(); x++) {
                    pool.emplace_back(interfaces[i + y][x + j]);
                }
            }
            clusters.emplace_back(pool);
        }
    }
}

void MaxPoolingLayer::forwardPropogate() {
    for(PoolingCluster& cluster : clusters) {
        cluster.forwardPropogate();
    }
}

void MaxPoolingLayer::backPropogate(const double learningRate) {
    for(PoolingCluster& cluster : clusters) {
        cluster.backPropogate();
    }
}

std::vector<NeuronInterface> MaxPoolingLayer::getInterfaces() {
    std::vector<NeuronInterface> interfaces;
    for(int i = 0; i < clusters.size(); i++) {
        interfaces.emplace_back(clusters[i].getInterface());
    }
    return interfaces;
}

std::vector<std::vector<NeuronInterface>> MaxPoolingLayer::getInterfaces2d() {
    std::vector<std::vector<NeuronInterface>> output;
    for(int i = 0; i < clusters.size(); i++) {
        if(i % rows == 0) {
            output.emplace_back();
        }
        output.back().push_back(clusters[i].getInterface());
    }
    return output;
}

MaxPoolingLayer::PoolingCluster::PoolingCluster(std::vector<NeuronInterface> pool): backInterfaces(pool), output(0), error(0) {
    
}

NeuronInterface MaxPoolingLayer::PoolingCluster::getInterface() {
    return NeuronInterface([this](double e){this->error += e;}, output);
}

void MaxPoolingLayer::PoolingCluster::forwardPropogate() {
    output = backInterfaces.front().output;
    lastMax = &backInterfaces.front();
    for(int i = 1; i < backInterfaces.size(); i++) {
        if(backInterfaces[i].output > output) {
            output = backInterfaces[i].output;
            lastMax = &backInterfaces[i];
        }
    }
}

void MaxPoolingLayer::PoolingCluster::backPropogate() {
    lastMax->errorAccumulator(error);
    error = 0;
}