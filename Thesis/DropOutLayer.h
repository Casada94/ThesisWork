#pragma once
#ifndef DROP_OUT_LAYER_H
#define DROP_OUT_LAYER_H
#include <vector>
#include <unordered_map>

class DropOutLayer {
private:
	DropOutLayer* previousLayer;
	DropOutLayer* nextLayer;
	int nodeCount;
	int activationFunctionSelected;
	int dropOutRate;
	bool isInputLayer;
	bool isOutputLayer;
	std::vector<double> output;
	std::vector<int> activeLayer;
	std::vector<std::vector<double>> weights;
	//weights	[ level in stack of prev layer's node ]
	//			[ level in stack of my layers node]
	//			[ weights from all the prev layers nodes to one of my nodes ]
	//			[ the node i am at now ]
	std::vector<double> bias;
	//std::unordered_map<std::string, double> backPropCache;

public:
	DropOutLayer(int nodeCount, int previousLayerNodeCount, int activationFunctionSelected, int dropOutRate, bool isInputLayer, bool isOutputLayer);
	void setPreviousLayer(DropOutLayer* previousLayer);
	void setNextLayer(DropOutLayer* nextLayer);
	void resetWeightsAndBias();
	void rollActiveLayers();
	void setOutput(std::vector<double>& rawInput);
	void setOutput(double rawInput);

	void updateAllWeights(double loss, double learningRate);
	void updateAllBiases(double loss, double learningRate);
	double getMyActPartDeriv(int index);
	double getPartDerivThrough(int fromNode, double loss);
	void getBackProp();

	void forwardPropagation();
	std::vector<double>& getOutput();
	std::vector<int>& getActiveLayer();
	double activationFunction(double sum);
};

#endif // DROP_OUT_LAYER_H