#pragma once
#ifndef FAT_LAYER_H
#define FAT_LAYER_H
#include <vector>
#include <random>
#include "Layer.h"

class FatLayer: public Layer {
private:
	int layerDepth;
	std::vector<std::vector<std::vector<std::vector<double>>>> weights;
	//weights	[ level in stack of prev layer's node ]
	//			[ level in stack of my layers node]
	//			[ weights from all the prev layers nodes to one of my nodes ]
	//			[ the node i am at now ]
	std::vector<std::vector<double>> bias;
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_int_distribution<int> distribution;
    std::uniform_int_distribution<int> distribution2;

public:
	FatLayer(int nodeCount, int previousLayerNodeCount, int layerDepth, int activationFunctionSelected,bool isInputLayer,bool isOutputLayer);
	void resetWeightsAndBias();
	void rollActiveLayers();
	
	void updateAllWeights(double loss, double learningRate);
	void updateAllBiases(double loss, double learningRate);
	double getPartDerivThrough(int fromNode, double loss);
	double getPartDerivThrough(int fromNode,int fromNodeStack, double loss);
	
	void forwardPropagation();
	void useAllNodes();
	void shakeWeights(double lowShake, double highShake);
	//std::vector<int>& getActiveLayer();
	//int getActiveLayerAt(int, int);
};

#endif // FAT_LAYER_H