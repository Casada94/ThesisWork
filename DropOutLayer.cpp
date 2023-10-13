#include "DropOutLayer.h"
#include <random>
#include <cmath>
#include <iostream>
#include <cstdlib>   // for rand() and srand() functions
#include <ctime>
#include <exception>

using namespace std;


//Constructor with minimum information needed to build layer
DropOutLayer::DropOutLayer(int nodeCount, int previousLayerNodeCount, int activationFunctionSelected, int dropOutRate, bool isInputLayer, bool isOutputLayer) {
	this->nodeCount = nodeCount;
	this->activationFunctionSelected = activationFunctionSelected;
	this->isInputLayer = isInputLayer;
	this->isOutputLayer = isOutputLayer;
	this->dropOutRate = dropOutRate;
	std::srand(static_cast<unsigned>(std::time(nullptr)));
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> distribution(1, 10000);


	if (!isInputLayer) {
		weights = std::vector<std::vector<double>>(previousLayerNodeCount, vector<double>(nodeCount, 0.0));
		bias = vector<double>(vector<double>(nodeCount, 0));
		activeLayer = vector<int>(nodeCount);
		output = vector<double>(nodeCount, 0);
		//std::srand(static_cast<unsigned>(std::time(nullptr)));
		for (int k = 0; k < previousLayerNodeCount; k++) {	//all the weights from T/B to my T/B
			for (int l = 0; l < nodeCount; l++) {
				weights[k][l] = (double)distribution(gen) / 10000;
				bias[l] = 0;
				if (k == 0) {
					int temp = std::rand() % 100;
					if (temp <= this->dropOutRate)
						activeLayer[l] = 0;
					else
						activeLayer[l] = 1;
				}
			}
		}


	}
	else {
		activeLayer = vector<int>(nodeCount);
		output = vector<double>(nodeCount, 0);
		for (int i = 0; i < nodeCount; i++) {
			int temp = std::rand() % 100;
			if (temp <= this->dropOutRate)
				activeLayer[i] = 0;
			else
				activeLayer[i] = 1;
		}
	}
}


//resets the weights and biases; used to reset the network for retraining
void DropOutLayer::resetWeightsAndBias() {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> distribution(1, 10000);

	int prevLayerNodeCount = previousLayer->getNodeCount();

	for (int k = 0; k < prevLayerNodeCount; k++) {
		for (int l = 0; l < nodeCount; l++) {	//all the weights from T/B to my T/B
			weights[k][l] = (double)distribution(gen) / 10000;
			if (k == 0)
				bias[k] = 0;
		}
	}
}

//randomly decides which subnode we will use in a 'fatNode'
void DropOutLayer::rollActiveLayers() {
	std::srand(static_cast<unsigned>(std::time(nullptr)));
	for (int i = 0; i < activeLayer.size(); i++) {
		int temp = std::rand() % 100;
		if (!isOutputLayer && temp <= this->dropOutRate)
			activeLayer[i] = 0;
		else
			activeLayer[i] = 1;
	}
}

void DropOutLayer::useAllNodes() {
	for (int i = 0; i < activeLayer.size(); i++) {
			activeLayer[i] = 1;
	}
}

//calculates the output of each 'node' and assigns the value to the property index of the output vector
void DropOutLayer::forwardPropagation() {
	double sum = 0;
	vector<double>& input = this->previousLayer->getOutput();
	for (int i = 0; i < nodeCount; i++) {
		sum = 0;
		if (!isOutputLayer) {
			for (int j = 0; j < input.size(); j++) {
				sum += weights[j][i] * input.at(j) * activeLayer[i];
			}
			sum += this->bias[i] * activeLayer[i];
		}
		else {
			for (int j = 0; j < input.size(); j++) {
				sum += weights[j][i] * input.at(j);
			}
			sum += this->bias[i];
		}
		sum = activationFunction(sum);
		//cout <<"SUM: " << sum << endl;
		this->output[i] = sum;
	}

}

void DropOutLayer::updateAllWeights(double loss, double learningRate) {
	//vector<int>& prevActiveLayer = previousLayer->getActiveLayer();
	vector<double>& prevOutput = previousLayer->getOutput();

	if (isOutputLayer) {
		for (int i = 0; i < nodeCount; i++) {
			for (int j = 0; j < prevOutput.size(); j++) {
				//this->weights[prevActiveLayer.at(j)][activeLayer[i]][j][i] -= prevOutput[j] * this->getMyActPartDeriv(i) * loss * learningRate;
				this->weights[j][i] -= prevOutput[j] * this->getMyActPartDeriv(i) * loss * learningRate;
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

void DropOutLayer::updateAllBiases(double loss, double learningRate) {
	for (int i = 0; i < nodeCount; i++) {
		if (this->isOutputLayer) {
			this->bias[i] -= this->getMyActPartDeriv(i) * loss * learningRate;
		}
		else {
			if (activeLayer[i])
				this->bias[i] -= this->getMyActPartDeriv(i) * nextLayer->getPartDerivThrough(i, loss) * learningRate;
		}

	}
}


//arg 'loss' is already the partialDeriv of loss function
double DropOutLayer::getPartDerivThrough(int fromNode, double loss) {
	double sum = 0.0;
	if (this->isOutputLayer) {
		for (int i = 0; i < nodeCount; i++) {
			sum += loss * this->getMyActPartDeriv(i) * this->weights[fromNode][i];
			//sum += this->getMyActPartDeriv(i); //* this->weights[fromNodeStack][this->activeLayer[i]][fromNode][i];
		}
	}
	else {
		for (int i = 0; i < nodeCount; i++) {
			//sum += getMyActPartDeriv(i) * previousLayer->getPartDerivThrough(i, activeLayer[i], loss);
			// ^^^^^^^^  I am not to sure about this
			if (activeLayer[i])
				sum += this->getMyActPartDeriv(i) * nextLayer->getPartDerivThrough(i, loss);

		}
	}
	return sum;
}
double DropOutLayer::getPartDerivThrough(int fromNode, int fromNodeStack, double loss) {
	throw std::runtime_error("Incorrect getPartDerivThrough() called for this class");
}
void DropOutLayer::shakeWeights(double lowShake, double highShake) {
	int prevLayerNodeCount = previousLayer->getNodeCount();
	std::srand(static_cast<unsigned>(std::time(nullptr)));

	for (int k = 0; k < prevLayerNodeCount; k++) {
		for (int l = 0; l < nodeCount; l++) {	//all the weights from T/B to my T/B
			if (rand() % 2) {
				weights[k][l] *= highShake;
			}
			else {
				weights[k][l] *= lowShake;
			}
		}
	}
}
//void DropOutLayer::getBackProp() {
//
//}


