#include "MultiLevelLayer.h"
#include <random>
#include <iostream>

using namespace std;

//Constructor with minimum information needed to build layer
MultiLevelLayer::MultiLevelLayer(int nodeCount, int levelSize, bool willUseAllNodes,bool isInputLayer,bool isOutputLayer) {
    this->isInputLayer = isInputLayer;
    this->isOutputLayer = isOutputLayer;
    this->levelSize = levelSize;
    this->trainingMode = true;

    this->willUseAllNodes = willUseAllNodes;
    this->scalar = (double)levelSize;

    output = vector<double>(nodeCount, 0);
    activeNodes = vector<int>(nodeCount,0);

    this->gen = std::mt19937(rd());
    this->distribution=std::uniform_int_distribution<int>(1, 10000);
    this->distribution2=std::uniform_int_distribution<int>(0, 99);

}

void MultiLevelLayer::forwardPropagation() {
    if(!this->isInputLayer){
        std::vector<double> prevOutput = this->previousLayer->getOutput();
        if(this->trainingMode && levelSize !=0 && willUseAllNodes){
            rollActiveNodes();
            for(int i=0; i < prevOutput.size(); i++) {
                output[i]= prevOutput[i] * activeNodes[i] * this->scalar;
            }
        } else if(!willUseAllNodes) {
            rollActiveNodes();
            for(int i=0; i < prevOutput.size(); i++) {
                output[i]= prevOutput[i] * activeNodes[i];
            }
        }
        else {
           for(int i=0; i < prevOutput.size(); i++) {
               output[i] = prevOutput[i];
           }
        }
    }
}

//randomly decides which subnode we will use in a 'fatNode'
void MultiLevelLayer::rollActiveNodes() {
    int temp=0;
    for (int i=0;i<activeNodes.size();i++) {
        if(i%levelSize==0){
            temp = i + (distribution2(gen) % levelSize);
        }
        activeNodes[i] = i==temp ? 1 : 0;
    }
}

double MultiLevelLayer::getPartDerivThrough(int fromNode, double loss){
    if(!isOutputLayer){
        if(activeNodes[fromNode] || levelSize==0){
            return nextLayer->getPartDerivThrough(fromNode, loss);
        }
        return 0.0;
    } else{
        return loss*activeNodes[fromNode];
    }
}


void MultiLevelLayer::setOutput(const std::vector<double>& rawInput) {
    if(rawInput.size() == output.size()){
        for(int i =0; i < this->output.size(); i++){
            output[i] = rawInput[i];
        }
    } else {
        throw std::runtime_error("Given match does not have the same size as output vector");
    }
}
std::vector<double>& MultiLevelLayer::getOutput() {
    if(isOutputLayer && output.size()!=1){
        output[0]+=output[1];;
    }
    return output;
}
