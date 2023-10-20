#pragma once
#ifndef LAYER_H
#define LAYER_H
#include <vector>
#include <unordered_map>

class Layer {
protected:
	Layer* previousLayer;
	Layer* nextLayer;
	int nodeCount;
	int activationFunctionSelected;
	bool isInputLayer;
	bool isOutputLayer;
	std::vector<double> output;
	std::vector<int>activeLayer;
	//weights	[ level in stack of prev layer's node ]
	//			[ level in stack of my layers node]
	//			[ weights from all the prev layers nodes to one of my nodes ]
	//			[ the node i am at now ]
	//std::unordered_map<std::string, double> backPropCache;

public:
	//Layer(int nodeCount, int previousLayerNodeCount, int activationFunctionSelected, bool isInputLayer, bool isOutputLayer);
	void setPreviousLayer(Layer* previousLayer);
	void setNextLayer(Layer* nextLayer);
	virtual void resetWeightsAndBias();
	virtual void rollActiveLayers();
	virtual void setOutput(std::vector<double>& rawInput);
	
	virtual void updateAllWeights(double loss, double learningRate);
	virtual void updateAllBiases(double loss, double learningRate);
	virtual double getMyActPartDeriv(int index);
	virtual double getPartDerivThrough(int fromNode, double loss);
	virtual double getPartDerivThrough(int fromNode, int fromNodeStack, double loss);

	virtual void forwardPropagation();
	std::vector<double>& getOutput();
	std::vector<int>& getActiveLayer();
	//virtual int getActiveLayerAt(int x,int y);
	double activationFunction(double sum);
	int getNodeCount();
	virtual void useAllNodes();
	virtual void shakeWeights(double lowShake, double highShake);
	//double getMyActPartDeriv(int index);
};

#endif // DROP_OUT_LAYER_H