//
// Created by metal on 10/29/2023.
//

#ifndef NETWORKEXPERIMENTS_DENSELAYER_H
#define NETWORKEXPERIMENTS_DENSELAYER_H
#include <vector>
#include <random>
#include "Layer.h"

class DenseLayer : public Layer{
protected:
    int nodeCount;
    int activationFunctionSelected;


    std::vector<std::vector<double>> weights;
    std::vector<double> bias;


public:
    DenseLayer(int nodeCount, int previousLayerNodeCount, int activationFunctionSelected, bool isInputLayer, bool isOutputLayer);
    void resetWeightsAndBias() override;
    void setOutput(const std::vector<double>& rawInput) override;

    void updateAllWeights(double loss, double learningRate) override;
    void updateAllBiases(double loss, double learningRate) override;
    double getMyActPartDeriv(int index) override;
    double getPartDerivThrough(int fromNode, double loss) override;

    void forwardPropagation() override;
    std::vector<double>& getOutput() override;
    double activationFunction(double sum) const override;
    int getNodeCount() const override;
    void shakeWeightsAndBiases(double delta) override;
};


#endif //NETWORKEXPERIMENTS_DENSELAYER_H
