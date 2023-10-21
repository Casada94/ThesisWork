#pragma once
#ifndef DROP_OUT_LAYER_H
#define DROP_OUT_LAYER_H

#include "Layer.h"
#include <vector>
#include <random>

class DropOutLayer: public Layer{
private:
	int dropOutRate;

public:
	DropOutLayer(int nodeCount, int previousLayerNodeCount, int activationFunctionSelected, int dropOutRate, bool isInputLayer, bool isOutputLayer);
    void rollActiveLayers() override;
    void scaleWeights() override;
};

#endif // DROP_OUT_LAYER_H