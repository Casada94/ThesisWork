#pragma once
#ifndef MIXED_LAYER_H
#define MIXED_LAYER_H

#include "Layer.h"
#include <vector>
#include <random>

class GBDropoutLayer: public Layer{
private:
    std::vector<int> activeNodes;
    int groupSize;
    double scalar;
    bool willUseAllNodes;

public:
	GBDropoutLayer(int nodeCount, int groupSize,bool willUseAllNodes, bool isInputLayer,bool isOutputLayer);
    void setOutput(const std::vector<double>& rawInput) override;
    double getPartDerivThrough(int fromNode, double loss) override;
    void forwardPropagation() override;
    void rollActiveNodes();
};

#endif // MIXED_LAYER_H