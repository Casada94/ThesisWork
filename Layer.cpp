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
void Layer::setMode(bool mode) {
    this->trainingMode = mode;
}

//METHODS WITHOUT ACTUAL IMPLEMENTATIONS
void Layer::setOutput(const std::vector<double>& rawInput) {}
int Layer::getNodeCount() const {  }
std::vector<double>& Layer::getOutput() {return output;}
void Layer::forwardPropagation() {}
double Layer::activationFunction(double sum) const {}
void Layer::updateAllWeights(double loss, double learningRate) {}
void Layer::updateAllBiases(double loss, double learningRate) {}
double Layer::getPartDerivThrough(int fromNode, double loss) {}
double Layer::getMyActPartDeriv(int index) {}
void Layer::resetWeightsAndBias() {}
void Layer::useAllNodes() {}
void Layer::shakeWeightsAndBiases(double delta) {}
void Layer::rollActiveLayers() {}
void Layer::scaleWeights() {};

