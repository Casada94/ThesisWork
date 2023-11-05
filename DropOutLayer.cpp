#include "DropOutLayer.h"
#include <random>
#include <iostream>

using namespace std;

//Constructor with minimum information needed to build layer
DropOutLayer::DropOutLayer(int nodeCount, int dropOutRate, bool willUseAllNodes,bool isInputLayer,bool isOutputLayer) {
	this->isInputLayer = isInputLayer;
	this->isOutputLayer = isOutputLayer;
	this->dropOutRate = dropOutRate;
    this->trainingMode=true;

    this->scalar = 1.0/((double)(100.0-dropOutRate)/100.0);
    this->willUseAllNodes = willUseAllNodes;

    output = vector<double>(nodeCount, 0);
    activeNodes = vector<int>(nodeCount,1);

    this->gen = std::mt19937(rd());
    this->distribution=std::uniform_int_distribution<int>(1, 10000);
    this->distribution2=std::uniform_int_distribution<int>(0, 99);

}

void DropOutLayer::forwardPropagation() {
    if(!isInputLayer){
        std::vector<double> prevOutput = this->previousLayer->getOutput();
        if(this->trainingMode && dropOutRate !=0 && willUseAllNodes){
            rollActiveNodes();
            for(int i=0; i < prevOutput.size(); i++) {
                output[i]= prevOutput[i] * activeNodes[i] * this->scalar;
            }
        } else if(!willUseAllNodes) {
            rollActiveNodes();
            for(int i=0; i < prevOutput.size(); i++) {
                output[i]= prevOutput[i] * activeNodes[i];
            }
        } else {
            for(int i=0; i < prevOutput.size(); i++) {
                output[i] = prevOutput[i];
            }
        }
    }
}

void DropOutLayer::rollActiveNodes() {
    for(int & activeNode : activeNodes) {
        activeNode = (distribution2(gen) < this->dropOutRate) ? 0:1;
    }
}


double DropOutLayer::getPartDerivThrough(int fromNode, double loss){
    if(activeNodes[fromNode] || dropOutRate==0){
        return this->scalar*nextLayer->getPartDerivThrough(fromNode, loss);
    }
    return 0.0;
}

void DropOutLayer::setOutput(const std::vector<double>& rawInput){
    if(rawInput.size() == output.size()){
        for(int i =0; i < this->output.size(); i++){
            output[i] = rawInput[i];
        }
    } else {
        throw std::runtime_error("Given match does not have the same size as output vector");
    }
}


