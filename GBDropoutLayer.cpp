#include "GBDropoutLayer.h"
#include <random>
#include <iostream>

using namespace std;

//Constructor with minimum information needed to build layer
GBDropoutLayer::GBDropoutLayer(int nodeCount, int previousLayerNodeCount, int activationFunctionSelected, int groupSize, bool isInputLayer, bool isOutputLayer) {
	this->nodeCount = nodeCount;
	this->activationFunctionSelected = activationFunctionSelected;
	this->isInputLayer = isInputLayer;
	this->isOutputLayer = isOutputLayer;
	this->groupSize = groupSize;

    if(!isInputLayer && !isOutputLayer && nodeCount%groupSize != 0){
        throw std::runtime_error("nodeCount is not a multiple of groupSize");
    }

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
        for(int i=0;i<nodeCount;i+=groupSize){
            activeLayer[i+(distribution2(gen) % groupSize)] = 0;
        }
    }
}

//randomly decides which subnode we will use in a 'fatNode'
void GBDropoutLayer::rollActiveLayers() {
    if(!isOutputLayer){
        int temp=0;
        for (int i=0;i<activeLayer.size();i++) {
            if(i%groupSize==0){
                temp = i + distribution2(gen) % groupSize;
            }
            activeLayer[i] = i==temp ? 0 : 1;
        }
    }
}

void GBDropoutLayer::scaleWeights() {
    int prevLayerNodeCount = previousLayer->getNodeCount();
    double scalingFactor=((double)(groupSize-1)/groupSize);
    for (int l = 0; l < nodeCount; l++) {	//all the weights from T/B to my T/B
        for (int k = 0; k < prevLayerNodeCount; k++) {
            weights[k][l] *= scalingFactor;
        }
        bias[l] *= scalingFactor;
    }
}

