#include "Layer.h"
#include <stdexcept>
#include <cmath>

//METHODS WITH ACTUAL IMPLEMENTATIONS
void Layer::setPreviousLayer(Layer* prevLayer) {
	this->previousLayer = prevLayer;
}
void Layer::setNextLayer(Layer* nxtLayer) {
	this->nextLayer = nxtLayer;
}
void Layer::setOutput(const std::vector<double>& rawInput) {
	if (rawInput.size() == output.size()) {
		for (int i = 0; i < rawInput.size(); i++) {
			this->output[i] = rawInput[i];
		}
	}
	else {
		throw std::runtime_error("Input size does not match number of nodes");
	}
}

std::vector<int>& Layer::getActiveLayer() { return this->activeLayer; }
int Layer::getNodeCount() const { return nodeCount; }
std::vector<double>& Layer::getOutput() {
	return output;
}

void Layer::forwardPropagation(double scalingFactor) {
    double sum;
    std::vector<double>& input = previousLayer->getOutput();

    for(int i =0; i<nodeCount;i++){
        sum=0;
        if(activeLayer[i]){
            for (int j = 0; j < input.size(); j++) {
                sum += weights[j][i] * input.at(j) * scalingFactor ;
            }
            output[i] = activationFunction(sum+ bias[i]) ;
        } else{
            output[i]=0;
        }
    }
    if(isOutputLayer && output.size()!=1){
        for(int i=0; i< nodeCount;i++)
            output[0] += output[i];
    }
}
double Layer::activationFunction(double sum) const {
	switch (activationFunctionSelected) {
	case 0:
		return 1.0 / (1.0 + std::exp(-sum));
	case 1:
		return (std::exp(sum) - std::exp(-sum)) / (std::exp(sum) + std::exp(-sum));
	case 2:
        return sum < 0 ? 0.0 : sum;
	default:
		return sum;
	}
}

void Layer::updateAllWeights(double loss, double learningRate) {
    std::vector<double>& prevOutput = previousLayer->getOutput();

    if (isOutputLayer) {
        for (int i = 0; i < nodeCount; i++) {
            if (activeLayer[i]) {
                for (int j = 0; j < prevOutput.size(); j++) {
                    this->weights[j][i] -= prevOutput[j] * this->getMyActPartDeriv(i) * loss * learningRate;
                }
            }
        }
    }
    else {
        for (int i = 0; i < nodeCount; i++) {
            if (activeLayer[i]) {
                for (int j = 0; j < prevOutput.size(); j++) {
                    this->weights[j][i] -= prevOutput[j] * nextLayer->getPartDerivThrough(i, loss) * this->getMyActPartDeriv(i) * learningRate;
                }
            }
        }
    }
}
void Layer::updateAllBiases(double loss, double learningRate) {
    for (int i = 0; i < nodeCount; i++) {
        if(activeLayer[i]){
            if (isOutputLayer) {
                bias[i] -= getMyActPartDeriv(i) * loss * learningRate;
            }
            else {
                bias[i] -= getMyActPartDeriv(i) * nextLayer->getPartDerivThrough(i, loss) * learningRate;
            }
        }
    }
}
double Layer::getPartDerivThrough(int fromNode, double loss) {
    double sum = 0.0;
    if (isOutputLayer) {
        for (int i = 0; i < nodeCount; i++) {
            sum += loss * getMyActPartDeriv(i) * weights[fromNode][i];
        }
    }
    else {
        for (int i = 0; i < nodeCount; i++) {
            if (activeLayer[i])
                sum += getMyActPartDeriv(i) * nextLayer->getPartDerivThrough(i, loss);

        }
    }
    return sum;
}
double Layer::getMyActPartDeriv(int index) {
    switch (activationFunctionSelected) {
        case 0:
            return output[index] * (1 - output[index]);
        case 1:
            return 1 - std::pow(output[index], 2);
        case 2:
            return output[index] <= 0 ? 0.0 : 1.0;
        default:
            return 1.0;
    }
}

void Layer::resetWeightsAndBias() {
    if(!isInputLayer){
        int prevLayerNodeCount = previousLayer->getNodeCount();

        for (int l = 0; l < nodeCount; l++) {	//all the weights from T/B to my T/B
            for (int k = 0; k < prevLayerNodeCount; k++) {
                weights[k][l] = (double)distribution(gen) / 10000;
            }
            bias[l] = 0;
        }
    }
}
void Layer::useAllNodes() {
    for (int &i: activeLayer)
        i = 1;
}
void Layer::shakeWeightsAndBiases(double delta) {
    if(!isInputLayer){
        int prevLayerNodeCount = previousLayer->getNodeCount();

        for (int l = 0; l < nodeCount; l++) {	//all the weights from T/B to my T/B
            for (int k = 0; k < prevLayerNodeCount; k++) {
                weights[k][l] = distribution2(gen)%2 ? weights[k][l]*(1-delta): weights[k][l]*(1+delta);
            }
            bias[l] = distribution2(gen)%2 ? bias[l]*(1-delta): bias[l]*(1+delta);
        }
    }
}

//METHODS WITHOUT ACTUAL IMPLEMENTATIONS
void Layer::rollActiveLayers() {}
void Layer::scaleWeights() {};

