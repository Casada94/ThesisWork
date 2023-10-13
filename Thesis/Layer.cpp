#include "Layer.h"

void Layer::setPreviousLayer(Layer* previousLayer) {
	this->previousLayer = previousLayer;
}

void Layer::setNextLayer(Layer* nextLayer) {
	this->nextLayer = nextLayer;
}

void Layer::setOutput(std::vector<double>& rawInput) {
	if (rawInput.size() == output.size()) {
		for (int i = 0; i < rawInput.size(); i++) {
			this->output[i] = rawInput[i];
		}
	}
	else {
		throw std::exception("Input size does not match number of nodes");
	}
}

std::vector<double>& Layer::getOutput() {
	return this->output;
}


double Layer::activationFunction(double sum) {
	switch (this->activationFunctionSelected) {
	case 0:
		return 1.0 / (1.0 + std::exp(-sum));
		break;
	case 1:
		return (std::exp(sum) - std::exp(-sum)) / (std::exp(sum) + std::exp(-sum));
		break;
	case 2:
		if (sum < 0) {
			return 0.0;
		}
		else {
			return sum;
		}
		break;
	default:
		return sum;
		//cerr << "Invalid activation function selected" << endl;
	}
}

double Layer::getMyActPartDeriv(int index) {
	switch (this->activationFunctionSelected) {
	case 0:
		return this->output[index] * (1 - this->output[index]);
		break;
	case 1:
		return 1 - pow(this->output[index], 2);
		break;
	case 2:
		if (this->output[index] <= 0) {
			return 0.0;
		}
		else {
			return 1.0;
		}
		break;
	default:
		return 1.0;
		//cerr << "Invalid activation function selected" << endl;
	}
}


void Layer::resetWeightsAndBias() {}
void Layer::rollActiveLayers() {}

void Layer::updateAllWeights(double loss, double learningRate) {}
void Layer::updateAllBiases(double loss, double learningRate) {}
double Layer::getPartDerivThrough(int fromNode, double loss) { return 0; }
double Layer::getPartDerivThrough(int fromNode, int fromNodeStack, double loss) { return 0; }

void Layer::forwardPropagation() {}
std::vector<int>& Layer::getActiveLayer() { return this->activeLayer; }
int Layer::getNodeCount() { return nodeCount; }
void Layer::useAllNodes() {}
void Layer::shakeWeights(double lowShake, double highShake) {}
