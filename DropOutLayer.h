#pragma once
#ifndef DROP_OUT_LAYER_H
#define DROP_OUT_LAYER_H
#include <vector>
#include <random>
#include "Layer.h"
//#include <unordered_map>

class DropOutLayer: public Layer{
private:
	int dropOutRate;
	std::vector<std::vector<double>> weights;
	//weights	[ level in stack of prev layer's node ]
	//			[ level in stack of my layers node]
	//			[ weights from all the prev layers nodes to one of my nodes ]
	//			[ the node i am at now ]
	std::vector<double> bias;
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_int_distribution<int> distribution;
    std::uniform_int_distribution<int> distribution2;

public:
	DropOutLayer(int nodeCount, int previousLayerNodeCount, int activationFunctionSelected, int dropOutRate, bool isInputLayer, bool isOutputLayer);
	//void setPreviousLayer(DropOutLayer* previousLayer);
	//void setNextLayer(DropOutLayer* nextLayer);
	void resetWeightsAndBias();
	void useAllNodes();
	void rollActiveLayers();
	//void setOutput(std::vector<double>& rawInput);
	//void setOutput(double rawInput);

	void updateAllWeights(double loss, double learningRate);
	void updateAllBiases(double loss, double learningRate);
	double getPartDerivThrough(int fromNode, double loss);
	double getPartDerivThrough(int fromNode, int fromNodeStack, double loss);
	void forwardPropagation();
	void shakeWeights(double lowShake, double highShake);
	//int getActiveLayerAt(int,int);
	
};

#endif // DROP_OUT_LAYER_H