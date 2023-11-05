#pragma once
#ifndef DROP_OUT_LAYER_H
#define DROP_OUT_LAYER_H

#include "Layer.h"
#include <vector>
#include <random>

class DropOutLayer: public Layer{
private:
    std::vector<int> activeNodes;
	int dropOutRate;
    double scalar;
    bool willUseAllNodes;

public:
	DropOutLayer(int nodeCount, int dropOutRate,bool willUseAllNodes,bool isInputLayer,bool isOutputLayer);
    void setOutput(const std::vector<double>& rawInput) override;
    double getPartDerivThrough(int fromNode, double loss) override;
    void forwardPropagation() override;
    void rollActiveNodes();
};

#endif // DROP_OUT_LAYER_H