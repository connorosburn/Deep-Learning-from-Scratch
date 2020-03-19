void Layer::initializeNeurons(std::vector<std::reference_wrapper<double>>  outputs, std::vector<std::reference_wrapper<double>> errors, int size) {
    for(int i = 0; i < size; i++) {
        neurons.emplace_back(outputs, errors);
    }
}