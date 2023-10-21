#include "DropOutLayer.h"
#include <random>
#include <iostream>

using namespace std;

//Constructor with minimum information needed to build layer
DropOutLayer::DropOutLayer(int nodeCount, int previousLayerNodeCount, int activationFunctionSelected, int dropOutRate, bool isInputLayer, bool isOutputLayer) {
	this->nodeCount = nodeCount;
	this->activationFunctionSelected = activationFunctionSelected;
	this->isInputLayer = isInputLayer;
	this->isOutputLayer = isOutputLayer;
	this->dropOutRate = dropOutRate;

    output = vector<double>(nodeCount, 0);
    activeLayer = vector<int>(nodeCount,1);

    this->gen = std::mt19937(rd());
    this->distribution=std::uniform_int_distribution<int>(1, 10000);
    this->distribution2=std::uniform_int_distribution<int>(0, 99);

    if (!isInputLayer) {
		weights = std::vector<std::vector<double>>(previousLayerNodeCount, vector<double>(nodeCount, 0.0));
        bias = vector<double>(nodeCount, 0);

        for (int k = 0; k < previousLayerNodeCount; k++) {	//all the weights from T/B to my T/B
			for (int l = 0; l < nodeCount; l++) {
				weights[k][l] = (double)distribution(gen) / 10000;
			}
		}
	}

    if(!isInputLayer && !isOutputLayer){
        for (int & i : activeLayer) {
            i = (distribution2(gen) < this->dropOutRate) ? 0:1;
        }
    }
}

//iterates through array marking nodes as active/inactive given a inactive probability
void DropOutLayer::rollActiveLayers() {
    if(!isOutputLayer){
        for (int & i : activeLayer) {
            i = (distribution2(gen) < this->dropOutRate) ? 0:1;
        }
    }
}

void DropOutLayer::scaleWeights(){
    int prevLayerNodeCount = previousLayer->getNodeCount();

    for (int l = 0; l < nodeCount; l++) {	//all the weights from T/B to my T/B
        for (int k = 0; k < prevLayerNodeCount; k++) {
            weights[k][l] *= 1-(double)(dropOutRate/100);
        }
        bias[l] *= 1-(double)(dropOutRate/100);
    }
}


