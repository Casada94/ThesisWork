#pragma once
#ifndef LAYER_H
#define LAYER_H
#include <vector>
#include <random>

class Layer {
protected:
	Layer* previousLayer;
	Layer* nextLayer;
	int nodeCount;
	int activationFunctionSelected;
	bool isInputLayer;
	bool isOutputLayer;

    std::vector<std::vector<double>> weights;
    std::vector<double> bias;
    std::vector<double> output;
	std::vector<int>activeLayer;

    std::random_device rd;
    std::mt19937 gen;
    std::uniform_int_distribution<int> distribution;
    std::uniform_int_distribution<int> distribution2;

public:
	//Layer(int nodeCount, int previousLayerNodeCount, int activationFunctionSelected, bool isInputLayer, bool isOutputLayer);
	void setPreviousLayer(Layer* prevLayer);
	void setNextLayer(Layer* nxtLayer);
	void resetWeightsAndBias();
	virtual void rollActiveLayers();
	virtual void setOutput(std::vector<double>& rawInput);
	
	virtual void updateAllWeights(double loss, double learningRate);
	void updateAllBiases(double loss, double learningRate);
	double getMyActPartDeriv(int index);
	double getPartDerivThrough(int fromNode, double loss);
//	double getPartDerivThrough(int fromNode, int fromNodeStack, double loss);

	void forwardPropagation();
	std::vector<double>& getOutput();
	std::vector<int>& getActiveLayer();
	double activationFunction(double sum) const;
	int getNodeCount() const;
	void useAllNodes();
    virtual void scaleWeights();
	void shakeWeightsAndBiases(double delta);
};

#endif // DROP_OUT_LAYER_H