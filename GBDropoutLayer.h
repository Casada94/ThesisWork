#pragma once
#ifndef MIXED_LAYER_H
#define MIXED_LAYER_H

#include "Layer.h"
#include <vector>
#include <random>

class GBDropoutLayer: public Layer{
private:
    int groupSize;

public:
	GBDropoutLayer(int nodeCount, int previousLayerNodeCount, int activationFunctionSelected, int groupSize, bool isInputLayer, bool isOutputLayer);
	void rollActiveLayers() override;
    void scaleWeights() override;
};

#endif // MIXED_LAYER_H