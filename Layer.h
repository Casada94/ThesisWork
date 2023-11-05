#pragma once
#ifndef LAYER_H
#define LAYER_H
#include <vector>
#include <random>

class Layer {
protected:
	Layer* previousLayer;
	Layer* nextLayer;
    bool isInputLayer;
	bool isOutputLayer;
    bool trainingMode;
    std::vector<double> output;

    std::random_device rd;
    std::mt19937 gen;
    std::uniform_int_distribution<int> distribution;
    std::uniform_int_distribution<int> distribution2;


public:
	//Layer(int nodeCount, int previousLayerNodeCount, int activationFunctionSelected, bool isInputLayer, bool isOutputLayer);
	void setPreviousLayer(Layer* prevLayer);
	void setNextLayer(Layer* nxtLayer);
    void setMode(bool);
    virtual void resetWeightsAndBias();
	virtual void rollActiveLayers();
	virtual void setOutput(const std::vector<double>& rawInput);

	virtual void updateAllWeights(double loss, double learningRate);
	virtual void updateAllBiases(double loss, double learningRate);
	virtual double getMyActPartDeriv(int index);
	virtual double getPartDerivThrough(int fromNode, double loss);
//	double getPartDerivThrough(int fromNode, int fromNodeStack, double loss);

	virtual void forwardPropagation();
	virtual std::vector<double>& getOutput();
	virtual double activationFunction(double sum) const;
	virtual int getNodeCount() const;
	virtual void useAllNodes();
    virtual void scaleWeights();
    virtual void shakeWeightsAndBiases(double delta);
};

#endif // DROP_OUT_LAYER_H