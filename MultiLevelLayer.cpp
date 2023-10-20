#include "MultiLevelLayer.h"
#include <random>
#include <iostream>

using namespace std;

//Constructor with minimum information needed to build layer
MultiLevelLayer::MultiLevelLayer(int nodeCount, int previousLayerNodeCount, int activationFunctionSelected, int levelSize, bool isInputLayer, bool isOutputLayer) {
    this->nodeCount = nodeCount*levelSize;
    this->activationFunctionSelected = activationFunctionSelected;
    this->isInputLayer = isInputLayer;
    this->isOutputLayer = isOutputLayer;
    this->levelSize = levelSize;


    output = vector<double>(this->nodeCount, 0);
    activeLayer = vector<int>(this->nodeCount,0);

    this->gen = std::mt19937(rd());
    this->distribution=std::uniform_int_distribution<int>(1, 10000);
    this->distribution2=std::uniform_int_distribution<int>(0, 100);

    if (!isInputLayer) {
        weights = std::vector<std::vector<double>>(previousLayerNodeCount, vector<double>(this->nodeCount, 0.0));
        bias = vector<double>(this->nodeCount, 0);

        for (int k = 0; k < previousLayerNodeCount; k++) {	//all the weights from T/B to my T/B
            for (int l = 0; l < this->nodeCount; l++) {
                weights[k][l] = (double)distribution(gen) / 10000;
            }
        }
    }

    for(int i=0;i<this->nodeCount;i+=levelSize){
        activeLayer[i+(distribution2(gen) % levelSize)] = 1;
    }
}

//randomly decides which subnode we will use in a 'fatNode'
void MultiLevelLayer::rollActiveLayers() {
    int temp=0;
    for (int i=0;i<activeLayer.size();i++) {
        if(i%levelSize==0){
            temp = i + distribution(gen) % levelSize;
        }
        activeLayer[i] = i==temp ? 1 : 0;
    }
}

void MultiLevelLayer::setOutput(std::vector<double>& input) {
    if(nodeCount/input.size() != levelSize){
        throw std::runtime_error("input size and input layer size are incompatible");
    }
    int inputIndex=0;
    for(int i=0;i<nodeCount;i++){
        output[i] = input[inputIndex];
        if(i%levelSize==levelSize-1){
            inputIndex++;
        }
    }
}