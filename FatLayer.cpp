#include "FatLayer.h"
#include <random>
#include <iostream>
#include <cstdlib>   // for rand() and srand() functions
#include <ctime>

using namespace std;


//Constructor with minimum information needed to build layer
FatLayer::FatLayer(int nodeCount, int previousLayerNodeCount, int layerDepth, int activationFunctionSelected, bool isInputLayer, bool isOutputLayer) {
	this->nodeCount = nodeCount;
	this->layerDepth = layerDepth;
	this->activationFunctionSelected = activationFunctionSelected;
	this->isInputLayer = isInputLayer;
	this->isOutputLayer = isOutputLayer;
	
	std::srand(static_cast<unsigned>(std::time(nullptr)));
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> distribution(1, 10000);


	if (!isInputLayer) {
		weights = std::vector < std::vector<std::vector<std::vector<double>>>>(layerDepth, vector < vector<vector<double>>>(layerDepth, vector < vector<double>>(previousLayerNodeCount, vector<double>(nodeCount,0.0))));
		bias = vector<vector<double>>(layerDepth, vector<double>(nodeCount, 0));
		this->activeLayer = vector<int>(nodeCount);
		output = vector<double>(nodeCount, 0);
		
		for (int i = 0; i < layerDepth; i++) {  //from top or bottom
			for (int j = 0; j < layerDepth; j++) {	//my top of bottom
				for (int k = 0; k < previousLayerNodeCount; k++) {	//all the weights from T/B to my T/B
					for (int l = 0; l < nodeCount; l++) {
						double temp = (double)distribution(gen) / 10000;
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
			this->activeLayer[i] = std::rand() % (layerDepth);
		}
	}
}

//resets the weights and biases; used to reset the network for retraining
void FatLayer::resetWeightsAndBias() {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> distribution(1, 10000);

	int prevLayerNodeCount = this->previousLayer->getNodeCount();

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

//calculates the output of each 'node' and assigns the value to the property index of the output vector
void FatLayer::forwardPropagation() {
	double sum = 0;
	vector<int>& prevActiveLayer = this->previousLayer->getActiveLayer();
	vector<double>& input = this->previousLayer->getOutput();
	for (int i = 0; i < nodeCount; i++) {
		sum = 0;
		for (int j = 0; j < input.size(); j++) {
			//cout << prevLayerActive.at(j) << ":" << this->activeLayer[i] <<":"  << j<<":" <<i << endl;
			//cout << weights[prevLayerActive.at(j)][this->activeLayer[i]][j][i] << endl;
			sum += weights[prevActiveLayer[j]][activeLayer[i]][j][i] * input.at(j);
		}
		sum += this->bias[this->activeLayer[i]][i];
		sum = activationFunction(sum);
		//cout <<"SUM: " << sum << endl;
		this->output[i] = sum;
	}

}


void FatLayer::updateAllWeights(double loss, double learningRate) {
	//vector<int>& prevActiveLayer = previousLayer->getActiveLayer();
	vector<double>& prevOutput = previousLayer->getOutput();
	vector<int>& prevActiveLayer = this->previousLayer->getActiveLayer();
	if (isOutputLayer) {
		for (int i = 0; i < nodeCount; i++) {
			for (int j = 0; j < prevOutput.size(); j++) {
				//this->weights[prevActiveLayer.at(j)][activeLayer[i]][j][i] -= prevOutput[j] * this->getMyActPartDeriv(i) * loss * learningRate;
				this->weights[prevActiveLayer[j]][activeLayer[i]][j][i] -= prevOutput[j] * this->getMyActPartDeriv(i) * loss * learningRate;
			}
		}
	}
	else {
		for (int i = 0; i < nodeCount; i++) {
			for (int j = 0; j < prevOutput.size(); j++) {
				//double temp = prevOutput[j] * nextLayer->getPartDerivThrough(i, activeLayer[i], loss) * this->getMyActPartDeriv(i) * learningRate;
				//double temp = nextLayer->getPartDerivThrough(i, activeLayer[i], loss) * learningRate;
				//cout << "adjustment: " << temp << endl;
				this->weights[prevActiveLayer[j]][activeLayer[i]][j][i] -= prevOutput[j] * nextLayer->getPartDerivThrough(i, activeLayer[i], loss) * this->getMyActPartDeriv(i) * learningRate;
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

double FatLayer::getPartDerivThrough(int fromNode, double loss) {
	throw std::runtime_error("Incorrect getPartDerivThrough() called for this class");
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

void FatLayer::useAllNodes() {
	throw std::runtime_error("This Layer type does not implement useAllNodes()");
}

void FatLayer::shakeWeights(double lowShake, double highShake) {
	int prevLayerNodeCount = this->previousLayer->getNodeCount();
	std::srand(static_cast<unsigned>(std::time(nullptr)));

	for (int i = 0; i < this->layerDepth; i++) {  //from top or bottom
		for (int j = 0; j < this->layerDepth; j++) {	//my top of bottom
			for (int k = 0; k < prevLayerNodeCount; k++) {
				for (int l = 0; l < this->nodeCount; l++) {	//all the weights from T/B to my T/B
					if (rand() % 2) {
						weights[i][j][k][l] *= highShake;
					}
					else {
						weights[i][j][k][l] *= lowShake;
					}
				}
			}
		}
	}
}