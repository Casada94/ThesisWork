#include "MultiLevelLayer.h"
#include <random>
#include <cmath>
#include <iostream>
#include <cstdlib>   // for rand() and srand() functions
#include <ctime>
#include <exception>

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
    bias = vector<double>(this->nodeCount, 0);

    this->gen = std::mt19937(rd());
    this->distribution=std::uniform_int_distribution<int>(1, 10000);
    this->distribution2=std::uniform_int_distribution<int>(0, 100);

    if (!isInputLayer) {
        weights = std::vector<std::vector<double>>(previousLayerNodeCount, vector<double>(this->nodeCount, 0.0));

        for (int k = 0; k < previousLayerNodeCount; k++) {	//all the weights from T/B to my T/B
            for (int l = 0; l < this->nodeCount; l++) {
                weights[k][l] = (double)distribution(gen) / 10000;
            }
        }
    }

//    if(!isInputLayer && !isOutputLayer){
        for(int i=0;i<this->nodeCount;i+=levelSize){
            activeLayer[i+(distribution2(gen) % levelSize)] = 1;
        }
//    }
}

//resets the weights and biases; used to reset the network for retraining
void MultiLevelLayer::resetWeightsAndBias() {
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
void MultiLevelLayer::rollActiveLayers() {
    int temp=0;
    for (int i=0;i<activeLayer.size();i++) {
        if(i%levelSize==0){
            temp = i + distribution(gen) % levelSize;
        }
        activeLayer[i] = i==temp ? 1 : 0;
    }
}

void MultiLevelLayer::useAllNodes() {
    for (int & i : activeLayer) {
        i = 1;
    }
}

//calculates the output of each 'node' and assigns the value to the property index of the output vector
void MultiLevelLayer::forwardPropagation() {
    double sum = 0;
    vector<double>& input = this->previousLayer->getOutput();

    for(int i =0; i<nodeCount;i++){
        sum=0;
        if(activeLayer[i]){
            for (int j = 0; j < input.size(); j++) {
                sum += weights[j][i] * input.at(j);
            }
            output[i] = activationFunction(sum + this->bias[i]) ;
        } else{
            output[i]=0;
            continue;
        }
    }
    if(isOutputLayer){
        output[0] = output[0] + output[1];
    }
}

void MultiLevelLayer::updateAllWeights(double loss, double learningRate) {
    vector<double>& prevOutput = previousLayer->getOutput();

    if (isOutputLayer) {
        for (int i = 0; i < nodeCount; i++) {
            if (activeLayer[i]) {
                for (int j = 0; j < prevOutput.size(); j++) {
                    this->weights[j][i] -= prevOutput[j] * this->getMyActPartDeriv(i) * loss * learningRate;
                }
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

void MultiLevelLayer::updateAllBiases(double loss, double learningRate) {
    for (int i = 0; i < nodeCount; i++) {
        if (this->isOutputLayer) {
            if (activeLayer[i])
                this->bias[i] -= this->getMyActPartDeriv(i) * loss * learningRate;
        }
        else {
            if (activeLayer[i])
                this->bias[i] -= this->getMyActPartDeriv(i) * nextLayer->getPartDerivThrough(i, loss) * learningRate;
        }
    }
}

//arg 'loss' is already the partialDeriv of loss function
double MultiLevelLayer::getPartDerivThrough(int fromNode, double loss) {
    double sum = 0.0;
    if (this->isOutputLayer) {
        for (int i = 0; i < nodeCount; i++) {
            if (activeLayer[i])
                sum += loss * this->getMyActPartDeriv(i) * this->weights[fromNode][i];
        }
    }
    else {
        for (int i = 0; i < nodeCount; i++) {
            if (activeLayer[i])
                sum += this->getMyActPartDeriv(i) * nextLayer->getPartDerivThrough(i, loss);

        }
    }
    return sum;
}
double MultiLevelLayer::getPartDerivThrough(int fromNode, int fromNodeStack, double loss) {
    double sum = 0.0;
    if (this->isOutputLayer) {
        for (int i = 0; i < nodeCount; i++) {
            if (activeLayer[i])
                sum += loss * this->getMyActPartDeriv(i) * this->weights[fromNode][i];
        }
    }
    else {
        for (int i = 0; i < nodeCount; i++) {
            if (activeLayer[i])
                sum += this->getMyActPartDeriv(i) * nextLayer->getPartDerivThrough(i, loss);

        }
    }
    return sum;
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
void MultiLevelLayer::shakeWeights(double lowShake, double highShake) {
    int prevLayerNodeCount = previousLayer->getNodeCount();

    for (int k = 0; k < prevLayerNodeCount; k++) {
        for (int l = 0; l < nodeCount; l++) {	//all the weights from T/B to my T/B
            if (distribution(gen) >=50) {
                weights[k][l] *= highShake;
            }
            else {
                weights[k][l] *= lowShake;
            }
        }
    }
}


