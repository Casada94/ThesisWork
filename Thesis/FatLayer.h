#pragma once
#ifndef FAT_LAYER_H
#define FAT_LAYER_H
#include <vector>
#include <unordered_map>

class FatLayer {
private:
	FatLayer* previousLayer;
	FatLayer* nextLayer;
	int nodeCount;
	int layerDepth;
	int activationFunctionSelected;
	bool isInputLayer;
	bool isOutputLayer;
	std::vector<double> output;
	std::vector<int> activeLayer;
	std::vector<std::vector<std::vector<std::vector<double>>>> weights;
	//weights	[ level in stack of prev layer's node ]
	//			[ level in stack of my layers node]
	//			[ weights from all the prev layers nodes to one of my nodes ]
	//			[ the node i am at now ]
	std::vector<std::vector<double>> bias;
	std::unordered_map<std::string, double> backPropCache;
	
public:
	FatLayer(int nodeCount, int previousLayerNodeCount, int layerDepth, int activationFunctionSelected,bool isInputLayer,bool isOutputLayer);
	void setPreviousLayer(FatLayer* previousLayer);
	void setNextLayer(FatLayer* nextLayer);
	void resetCache();
	void resetWeightsAndBias();
	void rollActiveLayers();
	void setOutput(std::vector<double>& rawInput);
	void setOutput(double rawInput);

	void updateAllWeights(double loss, double learningRate);
	void updateAllBiases(double loss, double learningRate);
	double getMyActPartDeriv(int index);
	double getPartDerivThrough(int fromNode,int fromNodeStack, double loss);
	void getBackProp();

	void forwardPropagation();
	std::vector<double>& getOutput();
	std::vector<int>& getActiveLayer();
	double activationFunction(double sum);
};

#endif // FAT_LAYER_H