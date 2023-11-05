#include "GBDropoutLayer.h"
#include <random>
#include <iostream>

using namespace std;

GBDropoutLayer::GBDropoutLayer(int nodeCount, int groupSize,bool willUseAllNodes, bool isInputLayer,bool isOutputLayer) {
    this->isInputLayer = isInputLayer;
    this->isOutputLayer = isOutputLayer;
    this->groupSize = groupSize;
    this->trainingMode=true;

    this->willUseAllNodes = willUseAllNodes;
    if(groupSize != 0)
        this->scalar = 1.0/((double)(groupSize-1)/groupSize);

    output = vector<double>(nodeCount, 0);
    activeNodes = vector<int>(nodeCount,1);

    this->gen = std::mt19937(rd());
    this->distribution=std::uniform_int_distribution<int>(1, 10000);
    this->distribution2=std::uniform_int_distribution<int>(0, 99);

}

void GBDropoutLayer::forwardPropagation() {
    if(!this->isInputLayer){
        std::vector<double> prevOutput = this->previousLayer->getOutput();
        if(this->trainingMode && groupSize !=0 && willUseAllNodes){
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

void GBDropoutLayer::rollActiveNodes() {
    int loser=0;
    for(int i=0; i < activeNodes.size(); i++) {
        if(i%groupSize == 0){
            loser = i + (distribution2(gen) % groupSize);
        }
        activeNodes[i] = i==loser ? 0 : 1;
    }
}

double GBDropoutLayer::getPartDerivThrough(int fromNode, double loss) {
    if(activeNodes[fromNode] || groupSize==0){
        return this->scalar*nextLayer->getPartDerivThrough(fromNode, loss);
    }
    return 0.0;
}

void GBDropoutLayer::setOutput(const std::vector<double>& rawInput) {
    if(rawInput.size() == output.size()){
        for(int i =0; i < this->output.size(); i++){
            output[i] = rawInput[i];
        }
    } else {
        throw std::runtime_error("Given match does not have the same size as output vector");
    }
}




