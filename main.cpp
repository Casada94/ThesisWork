#include "DropOutLayer.h"
#include "DenseLayer.h"
#include "GBDropoutLayer.h"
#include "MultiLevelLayer.h"
#include "DataPrep.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>

//MultiLevelLayer Methods
std::vector<std::unique_ptr<Layer>> createMultiLevelNetwork(std::vector<int>& layerNodeCount, int levelSize);

//DropOutLayer Methods
std::vector<std::unique_ptr<Layer>> createDropOutNetwork(std::vector<int>& layerNodeCount, int dropOutRate);

//GBDropoutLayer Methods
std::vector<std::unique_ptr<Layer>> createGBDropOutNetwork(std::vector<int>& layerNodeCount, int groupSize);

//Shared
void testNetwork(std::vector<std::unique_ptr<Layer>>& network,const std::vector<std::vector<double>>& input,const std::vector<double>& yTrue,bool useAllNodesFlag,std::string networkName,std::string coutPrefix, std::string trainingFileName, std::string predictFileName);
double forwardProp(std::vector< std::unique_ptr<Layer>>& network,const std::vector<double>& data);
void backProp(std::vector<std::unique_ptr<Layer>>& network, double loss);
void setTrainingMode(std::vector<std::unique_ptr<Layer>>& network,bool trainingMode);
//void resetNetworkWeightsAndBiases(std::vector<std::unique_ptr<Layer>>& network);
//void scaleAllWeights(std::vector<std::unique_ptr<Layer>>& gbDropOutNetwork);
//
//void rollNodes(std::vector<std::unique_ptr<Layer>>& dropOutNetwork);
//void shakeNetworkWeightsAndBiases(std::vector<std::unique_ptr<Layer>>& network,double delta);

float trainTestSplit =0.5;
float trainValidSplit=0.5;

int epochs = 100;
double learningRate = .01;

bool useAllNodesMulti = true;
bool useAllNodesGBD = true;
bool useAllNodesDrop = true;

/*  sigmoid = 0, tanh = 1, relu = 2, none = 3  */
int inputLayerAF = 1;
int hiddenLayerAF = 0;
int outputLayerAF = 3;

int main() {
    std::vector<int> multiLevelLayerNodeCounts = { 8,30,15,1 };
    std::vector<int> gbDropOutLayerNodeCounts = {8,32,16,1};
    std::vector<int> dropOutLayerNodeCounts =	{ 8,32,16,1 };

	int dropOutPercent = 50;
    int groupSize = 2;
	int layerDepth = 5;

    std::vector<std::unique_ptr<Layer>> multiLevelNetwork = createMultiLevelNetwork(multiLevelLayerNodeCounts, layerDepth);
//    std::vector<std::unique_ptr<Layer>> gbDropOutNetwork = createGBDropOutNetwork(gbDropOutLayerNodeCounts, groupSize);
//    std::vector<std::unique_ptr<Layer>> dropOutNetwork = createDropOutNetwork(dropOutLayerNodeCounts, dropOutPercent);

	std::vector<std::vector<double>> input(20640,std::vector<double>(8,0));
	std::vector<double> yTrue(20640,0);
    readDataSet(R"(C:\Users\metal\Desktop\Bayesian Neural Network Research\NetworkExperiments\ThesisWork\input\housingData.csv)", input);
//    readDataSet(R"(C:\Users\metal\Desktop\Bayesian Neural Network Research\NetworkExperiments\ThesisWork\input\normHousingData.csv)", input);
	readDataSet(R"(C:\Users\metal\Desktop\Bayesian Neural Network Research\NetworkExperiments\ThesisWork\input\housingDataTrueY.csv)", yTrue);
	
	//ROUND 1
	shuffleRows(input, yTrue, yTrue.size());
//    (double)layerDepth-1.0
    testNetwork(multiLevelNetwork,input,yTrue,useAllNodesMulti,"MultiLevel Network", "     ML = ",
                R"(C:\Users\metal\Desktop\Bayesian Neural Network Research\NetworkExperiments\ThesisWork\output\multiLevel\trainingLoss1.csv)",
                R"(C:\Users\metal\Desktop\Bayesian Neural Network Research\NetworkExperiments\ThesisWork\output\multiLevel\predictionOutput1.csv)");
//    testNetwork(gbDropOutNetwork,input,yTrue,useAllNodesGBD,"Group Based Dropout Network","GBDO = ",
//                R"(C:\Users\metal\Desktop\Bayesian Neural Network Research\NetworkExperiments\ThesisWork\output\groupBased\trainingLoss1.csv)",
//                R"(C:\Users\metal\Desktop\Bayesian Neural Network Research\NetworkExperiments\ThesisWork\output\groupBased\predictionOutput1.csv)");
//    testNetwork(dropOutNetwork,input,yTrue,useAllNodesDrop,"Dropout Network","     DO = ",
//                R"(C:\Users\metal\Desktop\Bayesian Neural Network Research\NetworkExperiments\ThesisWork\output\dropOut\trainingLoss1.csv)",
//                R"(C:\Users\metal\Desktop\Bayesian Neural Network Research\NetworkExperiments\ThesisWork\output\dropOut\predictionOutput1.csv)");

}

// MultiLevel LAYER METHODS
std::vector<std::unique_ptr<Layer>> createMultiLevelNetwork(std::vector<int>& layerNodeCount, int levelSize) {
    std::vector<std::unique_ptr<Layer>> network;
    for (int i = 0; i < layerNodeCount.size(); i++) {
        if (i == 0) {
            network.push_back(std::unique_ptr<Layer>(new MultiLevelLayer(layerNodeCount.at(i),0,useAllNodesMulti, true,false)));
        }
        else if (i == layerNodeCount.size() - 1) {
            network.push_back(std::unique_ptr<Layer>(new DenseLayer(layerNodeCount[i],layerNodeCount[i-1],outputLayerAF,false,true)));
//            network.push_back(std::unique_ptr<Layer>(new DenseLayer(layerNodeCount[i],layerNodeCount[i-1],outputLayerAF,false,false)));
//            network.push_back(std::unique_ptr<Layer>(new MultiLevelLayer(layerNodeCount.at(i), levelSize,useAllNodesMulti, false,true)));
        }
        else {
            network.push_back(std::unique_ptr<Layer>(new DenseLayer(layerNodeCount[i],layerNodeCount[i-1],hiddenLayerAF,false,false)));
            network.push_back(std::unique_ptr<Layer>(new MultiLevelLayer(layerNodeCount.at(i), levelSize,useAllNodesMulti, false, false)));
        }
    }
    for (int i = 0; i < network.size(); i++) {
        if (i == 0) {
            network.at(i).get()->setNextLayer(network.at(i+1).get());
        }
        else if (i == network.size() - 1) {
            network.at(i).get()->setPreviousLayer(network.at(i - 1).get());
        }
        else {
            network.at(i).get()->setNextLayer(network.at(i + 1).get());
            network.at(i).get()->setPreviousLayer(network.at(i - 1).get());
        }
    }
    return network;
}

//GROUP BASED DROPOUT NETWORK
std::vector<std::unique_ptr<Layer>> createGBDropOutNetwork(std::vector<int>& layerNodeCount, int groupSize) {
    std::vector<std::unique_ptr<Layer>> network;
    for (int i = 0; i < layerNodeCount.size(); i++) {
        if (i == 0) {
            network.push_back(std::unique_ptr<Layer>(new GBDropoutLayer(layerNodeCount.at(i), 0, useAllNodesGBD, true, false)));
        }
        else if (i == layerNodeCount.size() - 1) {
            network.push_back(std::unique_ptr<Layer>(new DenseLayer(layerNodeCount[i],layerNodeCount[i-1],outputLayerAF,false,true)));
        }
        else {
            network.push_back(std::unique_ptr<Layer>(new DenseLayer(layerNodeCount[i],layerNodeCount[i-1],hiddenLayerAF,false,false)));
            network.push_back(std::unique_ptr<Layer>(new GBDropoutLayer(layerNodeCount.at(i), groupSize,useAllNodesGBD, false, false)));
        }
    }
    for (int i = 0; i < network.size(); i++) {
        if (i == 0) {
            network.at(i)->setNextLayer(network.at(i + 1).get());
        }
        else if (i == network.size() - 1) {
            network.at(i)->setPreviousLayer(network.at(i - 1).get());
        }
        else {
            network.at(i)->setNextLayer(network.at(i + 1).get());
            network.at(i)->setPreviousLayer(network.at(i - 1).get());
        }
    }
    return network;
}

////DROP OUT LAYER METHODS
std::vector<std::unique_ptr<Layer>> createDropOutNetwork(std::vector<int>& layerNodeCount, int dropOutRate) {
    std::vector<std::unique_ptr<Layer>> network;
    for (int i = 0; i < layerNodeCount.size(); i++) {
        if (i == 0) {
            network.push_back(std::unique_ptr<Layer>(new DropOutLayer(layerNodeCount.at(i), 0, useAllNodesDrop, true, false)));
        }
        else if (i == layerNodeCount.size() - 1) {
            network.push_back(std::unique_ptr<Layer>(new DenseLayer(layerNodeCount[i],layerNodeCount[i-1],outputLayerAF,false,true)));
        }
        else {
            network.push_back(std::unique_ptr<Layer>(new DenseLayer(layerNodeCount[i],layerNodeCount[i-1],hiddenLayerAF,false,false)));
            network.push_back(std::unique_ptr<Layer>(new DropOutLayer(layerNodeCount.at(i), dropOutRate, useAllNodesDrop, false, false)));
        }
    }
    for (int i = 0; i < network.size(); i++) {
        if (i == 0) {
            network.at(i)->setNextLayer(network.at(i + 1).get());
        }
        else if (i == network.size() - 1) {
            network.at(i)->setPreviousLayer(network.at(i - 1).get());
        }
        else {
            network.at(i)->setNextLayer(network.at(i + 1).get());
            network.at(i)->setPreviousLayer(network.at(i - 1).get());
        }
    }
    return network;
}

//SHARED METHODS
void testNetwork(std::vector<std::unique_ptr<Layer>>& network,const std::vector<std::vector<double>>& input,const std::vector<double>& yTrue,bool useAllNodesFlag,std::string networkName,std::string coutPrefix, std::string trainingFileName, std::string predictFileName) {
    std::ofstream trainingFile(trainingFileName);
    std::ofstream testingFile(predictFileName);
    std::cout << std::fixed << std::setprecision(7);
    trainingFile << std::fixed << std::setprecision(7);
    testingFile << std::fixed << std::setprecision(12);
    std::cout  << "----------"<< networkName <<"----------" << std::endl;
    char a = 177, b = 219;
    for(int i=0;i<50;i++){
        std::cout<<a;
    }
    int progression=epochs/50;
    std::cout<<"\r";


    double yHat = 0.0;
    double loss = 0;
    double trainingLoss = 0.0;
    double validLoss = 0.0;

    int startOfTestIndex = (int)(input.size() * trainTestSplit);
    int startOfValidIndex = (int)(startOfTestIndex * trainValidSplit);


    for (int i = 0; i < epochs; i++) {
        if(i%progression==0){
            std::cout<<b;
        }
        trainingLoss = 0;

        setTrainingMode(network,true);

        //TRAINING
        for (int j = 0; j < startOfValidIndex; j++) {
            yHat = forwardProp(network, input[j]);
            loss = 2 * (yHat - (yTrue[j]));
            trainingLoss += std::pow((yHat - (yTrue[j])), 2);
            backProp(network, loss);
        }

        validLoss = 0;
        setTrainingMode(network,false);
        for (int j = startOfValidIndex; j < startOfTestIndex; j++) {

            yHat = forwardProp(network, input[j]);
            validLoss += std::pow((yHat - yTrue[j]), 2);
        }

        trainingLoss /= startOfValidIndex;
        validLoss /= (startOfTestIndex - startOfValidIndex);
//        std::cout << "Epoch: " << i << "\t" << "Training Loss: " << trainingLoss << "\t" << "Valid Loss: " << validLoss << std::endl;
        trainingFile << trainingLoss << "," << validLoss << "\n";
    }

    loss = 0;

    std::vector<double> yHatMulti = std::vector<double>(3,0);
    double sum=0;
    double onceLoss = 0;
    yHat=0;
    //TESTING
    for (int j = startOfTestIndex; j < input.size(); j++) {
        for(int i=0;i<3;i++){
            yHatMulti[i] = forwardProp(network, input[j]);
            sum+=yHatMulti[i];
        }

        yHat = sum/3;
        loss += std::pow((yHat - yTrue[j]), 2);
        onceLoss += std::pow((yHatMulti[0] - yTrue[j]), 2);
        sum=0;
        testingFile << yHatMulti[0] << "," << yHatMulti[1] << "," << yHatMulti[2] << "," << yTrue[j] << "\n";
    }
//    std::cout << loss << std::endl;
//    std::cout << onceLoss << std::endl;
    std::cout <<"\r" << coutPrefix << "3 Pass Test Loss: " << loss / (input.size() - startOfTestIndex) << "      1 Pass Test Loss: " << onceLoss / (input.size() - startOfTestIndex) << std::endl;
    testingFile << loss << "," << onceLoss << "," << "00" << "," << "00"<< "\n";

    trainingFile.close();
    testingFile.close();
}
double forwardProp(std::vector< std::unique_ptr<Layer>>& network,const std::vector<double>& data) {
    for (int layer = 0; layer < network.size(); layer++) {
        if (layer == 0)
            network.at(layer).get()->setOutput(data);
//        else if(layer==network.size()-1)
//            network.at(layer).get()->forwardPropagation(1);
        else
            network.at(layer).get()->forwardPropagation();
    }
    return network.at(network.size() - 1).get()->getOutput()[0];
}
void backProp(std::vector<std::unique_ptr<Layer>>& network, double loss) {
    for (int layer = 1; layer < network.size(); layer++) {
        network.at(layer).get()->updateAllBiases(loss, learningRate);
        network.at(layer).get()->updateAllWeights(loss, learningRate);
    }
}

void setTrainingMode(std::vector<std::unique_ptr<Layer>>& network,bool trainingMode) {
    for (const auto & layer : network)
        layer->setMode(trainingMode);
}



