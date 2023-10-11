#include "FatLayer.h"
#include <random>
#include <cmath>
#include <iostream>
#include <cstdlib>   // for rand() and srand() functions
#include <ctime>
#include <exception>

using namespace std;


//Constructor with minimum information needed to build layer
FatLayer::FatLayer(int nodeCount, int previousLayerNodeCount, int layerDepth, int activationFunctionSelected, bool isInputLayer, bool isOutputLayer) {
	this->nodeCount = nodeCount;
	this->layerDepth = layerDepth;
	this->activationFunctionSelected = activationFunctionSelected;
	this->isInputLayer = isInputLayer;
	this->isOutputLayer = isOutputLayer;
	//std::srand(static_cast<unsigned>(std::time(nullptr)));
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> distribution(1, 10000);


	if (!isInputLayer) {
		weights = std::vector < std::vector<std::vector<std::vector<double>>>>(layerDepth, vector < vector<vector<double>>>(layerDepth, vector < vector<double>>(previousLayerNodeCount, vector<double>(nodeCount,0.0))));
		bias = vector<vector<double>>(layerDepth, vector<double>(nodeCount, 0));
		activeLayer = vector<int>(nodeCount);
		output = vector<double>(nodeCount, 0);
		//std::srand(static_cast<unsigned>(std::time(nullptr)));
		for (int i = 0; i < layerDepth; i++) {  //from top or bottom
			for (int j = 0; j < layerDepth; j++) {	//my top of bottom
				for (int k = 0; k < previousLayerNodeCount; k++) {	//all the weights from T/B to my T/B
					for (int l = 0; l < nodeCount; l++) {
						double temp = (double)distribution(gen) / 10000;
						//if (temp < 0.0001 || temp > 1) {
						//	throw std::runtime_error("An error occurred.");
						//}
						weights[i][j][k][l] = temp;
						bias[i][l] = 0;
						if (j == 0) {
							activeLayer[l] = std::rand() % (layerDepth);

						}
					}
				}
			}
		}
	}
	else {
		activeLayer = vector<int>(nodeCount);
		output = vector<double>(nodeCount, 0);
		for (int i = 0; i < nodeCount; i++) {
			activeLayer[i] = std::rand() % (layerDepth);
		}
	}
}

//Setter for previous layer pointer; needed for backpropagation
void FatLayer::setPreviousLayer(FatLayer* previousLayer) {
	this->previousLayer = previousLayer;
}

//setter for nextLayer needed for backpropagation
void FatLayer::setNextLayer(FatLayer* nextLayer) {
	this->nextLayer = nextLayer;
}

//resets the weights and biases; used to reset the network for retraining
void FatLayer::resetWeightsAndBias() {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> distribution(1, 10000);

	int prevLayerNodeCount = this->previousLayer->getActiveLayer().size();

	for (int i = 0; i < this->layerDepth; i++) {  //from top or bottom
		for (int j = 0; j < this->layerDepth; j++) {	//my top of bottom
			for (int k = 0; k < prevLayerNodeCount; k++) {
				for (int l = 0; l < this->nodeCount; l++) {	//all the weights from T/B to my T/B
					double temp = (double)distribution(gen) / 10000;
					if (temp < 0 || temp > 1) {
						throw std::runtime_error("An error occurred.");
					}
					weights[i][j][k][l] = temp;
					bias[i][l] = 0;
				}
			}
		}
	}
}

//randomly decides which subnode we will use in a 'fatNode'
void FatLayer::rollActiveLayers() {
	std::srand(static_cast<unsigned>(std::time(nullptr)));
	for (int i = 0; i < activeLayer.size(); i++) {
		activeLayer[i] = rand() % (layerDepth);
	}
}

void FatLayer::setOutput(vector<double>& rawInput) {
	if (this->isInputLayer && rawInput.size() == this->output.size()) {
		for (int i = 0; i < this->output.size(); i++) {
			this->output[i] = rawInput.at(i);
		}
	}
}
void FatLayer::setOutput(double rawInput) {
	if (this->isInputLayer) {
		this->output[0] = rawInput;
	}
}

//calculates the output of each 'node' and assigns the value to the property index of the output vector
void FatLayer::forwardPropagation() {
	double sum = 0;
	vector<double>& input = this->previousLayer->getOutput();
	vector<int>& prevLayerActive = this->previousLayer->getActiveLayer();
	for (int i = 0; i < nodeCount; i++) {
		sum = 0;
		for (int j = 0; j < input.size(); j++) {
			//cout << prevLayerActive.at(j) << ":" << this->activeLayer[i] <<":"  << j<<":" <<i << endl;
			//cout << weights[prevLayerActive.at(j)][this->activeLayer[i]][j][i] << endl;
			sum += weights[prevLayerActive.at(j)][activeLayer[i]][j][i] * input.at(j);
		}
		sum += this->bias[this->activeLayer[i]][i];
		sum = activationFunction(sum);
		//cout <<"SUM: " << sum << endl;
		this->output[i] = sum;
	}

}

vector<double>& FatLayer::getOutput() {
	return this->output;
}
vector<int>& FatLayer::getActiveLayer() {
	return this->activeLayer;
}

//calculated the output of a 'node' using the configured activation function
double FatLayer::activationFunction(double sum) {
	switch (this->activationFunctionSelected) {
	case 0:
		return 1.0 / (1.0 + std::exp(-sum));
		break;
	case 1:
		return (std::exp(sum) - std::exp(-sum)) / (std::exp(sum) + std::exp(-sum));
		break;
	case 2:
		if(sum < 0) {
			return 0.0;
		} else {
			return sum;
		}
		break;
	default:
		return sum;
		//cerr << "Invalid activation function selected" << endl;
	}
}

//clears the cache of previously calculated backpropagation values
//used as a dynamic programming solution to finding backpropagation values
void FatLayer::resetCache() {
	this->backPropCache.clear();
}


void FatLayer::updateAllWeights(double loss, double learningRate) {
	vector<int>& prevActiveLayer = previousLayer->getActiveLayer();
	vector<double>& prevOutput = previousLayer->getOutput();

	if (isOutputLayer) {
		for (int i = 0; i < nodeCount; i++) {
			for (int j = 0; j < prevActiveLayer.size(); j++) {
				//this->weights[prevActiveLayer.at(j)][activeLayer[i]][j][i] -= prevOutput[j] * this->getMyActPartDeriv(i) * loss * learningRate;
				this->weights[prevActiveLayer.at(j)][activeLayer[i]][j][i] -= prevOutput[j] * this->getMyActPartDeriv(i) * loss * learningRate;
			}
		}
	}
	else {
		for (int i = 0; i < nodeCount; i++) {
			for (int j = 0; j < prevActiveLayer.size(); j++) {
				//double temp = prevOutput[j] * nextLayer->getPartDerivThrough(i, activeLayer[i], loss) * this->getMyActPartDeriv(i) * learningRate;
				//double temp = nextLayer->getPartDerivThrough(i, activeLayer[i], loss) * learningRate;
				//cout << "adjustment: " << temp << endl;
				this->weights[prevActiveLayer.at(j)][activeLayer[i]][j][i] -= prevOutput[j] * nextLayer->getPartDerivThrough(i, activeLayer[i], loss) * this->getMyActPartDeriv(i) * learningRate;
			}
		}
	}
}

void FatLayer::updateAllBiases(double loss, double learningRate) {
	for (int i = 0; i < nodeCount; i++) {
		if (this->isOutputLayer) {
			this->bias[activeLayer[i]][i] -= this->getMyActPartDeriv(i) * loss * learningRate;
		}
		else {
			this->bias[activeLayer[i]][i] -= this->getMyActPartDeriv(i) * nextLayer->getPartDerivThrough(i, this->activeLayer[i], loss) * learningRate;
		}
		
	}
}

double FatLayer::getMyActPartDeriv(int index) {
	switch (this->activationFunctionSelected) {
	case 0:
		return this->output[index]* (1 - this->output[index]);
		break;
	case 1:
		return 1 - pow(this->output[index],2);
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

//arg 'loss' is already the partialDeriv of loss function
double FatLayer::getPartDerivThrough(int fromNode,int fromNodeStack, double loss) {
	double sum = 0.0;
	if (this->isOutputLayer) {
		for (int i = 0; i < nodeCount; i++) {
			sum += loss *this->getMyActPartDeriv(i)* this->weights[fromNodeStack][this->activeLayer[i]][fromNode][i];
			//sum += this->getMyActPartDeriv(i); //* this->weights[fromNodeStack][this->activeLayer[i]][fromNode][i];
		}
	}
	else {
		for (int i = 0; i < nodeCount; i++) {
			//sum += getMyActPartDeriv(i) * previousLayer->getPartDerivThrough(i, activeLayer[i], loss);
			// ^^^^^^^^  I am not to sure about this
			sum += this->getMyActPartDeriv(i) * nextLayer->getPartDerivThrough(i, this->activeLayer[i], loss);
			
		}
	}
	return sum;
}

void FatLayer::getBackProp() {
	
}


