#include "DropOutLayer.h"
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

//std::vector<std::unique_ptr<Layer>> createHybridNetwork(std::vector<int>& layerNodeCount, std::vector<int>& layerTypeDef, int layerDepth, int dropOutRate);

//MultiLevelLayer Methods
std::vector<std::unique_ptr<Layer>> createMultiLevelNetwork(std::vector<int>& layerNodeCount, int levelSize);
void testMultiLevelNetwork(std::vector<std::unique_ptr<Layer>>& multiLevelNetwork, std::vector<std::vector<double>>& input, std::vector<double>& yTrue, std::string trainingFile, std::string predictFile);

//DropOutLayer Methods
std::vector<std::unique_ptr<Layer>> createDropOutNetwork(std::vector<int>& layerNodeCount, int dropOutRate);
void testDropOutNetwork(std::vector<std::unique_ptr<Layer>>& dropOutNetwork, std::vector<std::vector<double>>& input,double scalingFactor, std::vector<double>& yTrue, std::string trainingFile, std::string predictFile);
void activateAllNodes(std::vector<std::unique_ptr<Layer>>& network);

//GBDropoutLayer Methods
std::vector<std::unique_ptr<Layer>> createGBDropOutNetwork(std::vector<int>& layerNodeCount, int groupSize);
void testGBDropOutNetwork(std::vector<std::unique_ptr<Layer>>& dropOutNetwork, std::vector<std::vector<double>>& input,double scalingFactor, std::vector<double>& yTrue, std::string trainingFile, std::string predictFile);

//Shared
std::vector<double> forwardProp(std::vector< std::unique_ptr<Layer>>& network, std::vector<double>& data, double scalingFactor);
void backProp(std::vector<std::unique_ptr<Layer>>& network, double loss);
void resetNetworkWeightsAndBiases(std::vector<std::unique_ptr<Layer>>& network);
void scaleAllWeights(std::vector<std::unique_ptr<Layer>>& gbDropOutNetwork);

void rollNodes(std::vector<std::unique_ptr<Layer>>& dropOutNetwork);
void shakeNetworkWeightsAndBiases(std::vector<std::unique_ptr<Layer>>& network,double delta);

float trainTestSplit =0.75;
float trainValidSplit=0.5;

int epochs = 100;
double learningRate = .01;

bool useAllNodesMulti = false;
bool useAllNodesGBD = true;
bool useAllNodesDrop = true;

bool scaleWeights = false;

/*  sigmoid = 0, tanh = 1, relu = 2, none = 3  */
int inputLayerAF = 1;
int hiddenLayerAF = 0;
int outputLayerAF = 3;

int main() {
    std::vector<int> multiLevelLayerNodeCounts = { 8,8,5,1 };
    std::vector<int> gbDropOutLayerNodeCounts = {8,15,10,1};
    std::vector<int> dropOutLayerNodeCounts =	{ 8,15,10,1 };

	int dropOutPercent = 20;
    int groupSize = 5;
	int layerDepth = 2;

    std::vector<std::unique_ptr<Layer>> multiLevelNetwork = createMultiLevelNetwork(multiLevelLayerNodeCounts, layerDepth);
    std::vector<std::unique_ptr<Layer>> gbDropOutNetwork = createGBDropOutNetwork(gbDropOutLayerNodeCounts, groupSize);
    std::vector<std::unique_ptr<Layer>> dropOutNetwork = createDropOutNetwork(dropOutLayerNodeCounts, dropOutPercent);

	std::vector<std::vector<double>> input(20640,std::vector<double>(8,0));
	std::vector<double> yTrue(20640,0);
	readDataSet("C:\\Users\\metal\\Desktop\\Bayesian Neural Network Research\\NetworkExperiments\\ThesisWork\\input\\housingData.csv", input);
	readDataSet("C:\\Users\\metal\\Desktop\\Bayesian Neural Network Research\\NetworkExperiments\\ThesisWork\\input\\housingDataTrueY.csv", yTrue);
	
	//ROUND 1
	shuffleRows(input, yTrue, yTrue.size());

	testFatNetwork(fatNetwork,epochs,learningRate,input,yTrue,
		"C:\\Users\\metal\\Desktop\\Bayesian Neural Network Research\\NetworkExperiments\\ThesisWork\\output\\fatNode\\trainingLoss1.csv",
		"C:\\Users\\metal\\Desktop\\Bayesian Neural Network Research\\NetworkExperiments\\ThesisWork\\output\\fatNode\\predictionOutput1.csv");
    testMixedNetwork(mixedNetwork, epochs, learningRate, input, yTrue,
                       "C:\\Users\\metal\\Desktop\\Bayesian Neural Network Research\\NetworkExperiments\\ThesisWork\\output\\mixed\\trainingLoss1.csv",
                       "C:\\Users\\metal\\Desktop\\Bayesian Neural Network Research\\NetworkExperiments\\ThesisWork\\output\\mixed\\predictionOutput1.csv");
    testDropOutNetwork(dropOutNetwork, epochs, learningRate, input, yTrue,
	testGBDropOutNetwork(gbDropOutNetwork, input, yTrue,
                       "C:\\Users\\metal\\Desktop\\Bayesian Neural Network Research\\NetworkExperiments\\ThesisWork\\output\\dropOut\\trainingLoss1.csv",
                       "C:\\Users\\metal\\Desktop\\Bayesian Neural Network Research\\NetworkExperiments\\ThesisWork\\output\\dropOut\\predictionOutput1.csv");


}

// MultiLevel LAYER METHODS
std::vector<std::unique_ptr<Layer>> createMultiLevelNetwork(std::vector<int>& layerNodeCount, int levelSize) {
    std::vector<std::unique_ptr<Layer>> network;
    for (int i = 0; i < layerNodeCount.size(); i++) {
        if (i == 0) {
            network.push_back(std::unique_ptr<Layer>(new MultiLevelLayer(layerNodeCount.at(i), 0, inputLayerAF,levelSize, true, false)));
        }
        else if (i == layerNodeCount.size() - 1) {
            network.push_back(std::unique_ptr<Layer>(new MultiLevelLayer(layerNodeCount.at(i), layerNodeCount.at(i-1)*levelSize, outputLayerAF, levelSize, false, true)));
        }
        else {
            network.push_back(std::unique_ptr<Layer>(new MultiLevelLayer(layerNodeCount.at(i), layerNodeCount.at(i - 1)*levelSize, hiddenLayerAF, levelSize, false, false)));
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
void testMultiLevelNetwork(std::vector<std::unique_ptr<Layer>>& multiLevelNetwork, std::vector<std::vector<double>>& input, std::vector<double>& yTrue,std::string trainingFile,std::string predictFile) {
    std::ofstream multiLevelLayerTrainingFile(trainingFile);
    std::ofstream multiLevelLayerTestingFile(predictFile);

    double yHat = 0.0;
    double loss = 0;
    double trainingLoss = 0.0;
    double validLoss = 0.0;

    int startOfTestIndex = (int)(input.size() * trainTestSplit);
    int startOfValidIndex = (int)(startOfTestIndex * trainValidSplit);

    std::cout << std::fixed << std::setprecision(7);
    multiLevelLayerTrainingFile << std::fixed << std::setprecision(7);
    multiLevelLayerTestingFile << std::fixed << std::setprecision(12);
    std::cout  << "----------Multi Level Network----------" << std::endl;

    for (int i = 0; i < epochs; i++) {
        trainingLoss = 0;

        for (int j = 0; j < startOfValidIndex; j++) {
            yHat = forwardProp(multiLevelNetwork, input[j],1)[0];
            loss = 2 * (yHat - yTrue[j]);
            trainingLoss += std::pow((yHat - yTrue[j]), 2);
            backProp(multiLevelNetwork, loss);
            rollNodes(multiLevelNetwork);
        }

        validLoss = 0;
        if(useAllNodesMulti){
            activateAllNodes(multiLevelNetwork);
            if(scaleWeights){
                scaleAllWeights(multiLevelNetwork);
            }
        }
        for (int j = startOfValidIndex; j < startOfTestIndex; j++) {
            if(!useAllNodesMulti){
                rollNodes(multiLevelNetwork);
            }
            yHat = forwardProp(multiLevelNetwork, input[j],1)[0];
            validLoss += std::pow((yHat - yTrue[j]), 2);
        }

        trainingLoss /= startOfValidIndex;
        validLoss /= (startOfTestIndex - startOfValidIndex);
        std::cout  << "Epoch: " << i << "\t" << "Training Loss: " << trainingLoss<< "\t" << "Valid Loss: " << validLoss << std::endl;
        multiLevelLayerTrainingFile  << trainingLoss << "," << validLoss  << "\n";
    }

    loss = 0;
    std::vector<double> yHatMult = std::vector<double>(3,0);
    double sum=0;
    double onceLoss = 0;
    yHat=0;
    if(useAllNodesMulti){
        if(scaleWeights){
            scaleAllWeights(multiLevelNetwork);
        }
    }
    for (int j = startOfTestIndex; j < input.size(); j++) {
        for(int i=0;i<3;i++){
            if(!useAllNodesMulti){
                rollNodes(multiLevelNetwork);
            }
            yHatMult[i] = forwardProp(multiLevelNetwork, input[j],1)[0];
            sum+=yHatMult[i];
        }

        yHat = sum/3;
        loss += std::pow((yHat - yTrue[j]), 2);
        onceLoss += std::pow((yHatMult[0] - yTrue[j]), 2);
        sum=0;
        multiLevelLayerTestingFile << yHatMult[0] << "," << yHatMult[1] << "," << yHatMult[2] << "," << yTrue[j] << "\n";
    }

    std::cout << loss << std::endl;
    std::cout << onceLoss << std::endl;
    std::cout << std::fixed << std::setprecision(7) << "     ML = 3 Pass Test Loss: " << loss / (input.size() - startOfTestIndex) << "\t" << "   1 Pass Test Loss: " << onceLoss / (input.size() - startOfTestIndex) << std::endl;
    multiLevelLayerTestingFile << loss << "," << onceLoss << "," << "00" << "," << "00" << "\n";

    multiLevelLayerTrainingFile.close();
    multiLevelLayerTestingFile.close();
}

//GROUP BASED DROPOUT NETWORK
std::vector<std::unique_ptr<Layer>> createGBDropOutNetwork(std::vector<int>& layerNodeCount, int groupSize) {
    std::vector<std::unique_ptr<Layer>> network;
    for (int i = 0; i < layerNodeCount.size(); i++) {
        if (i == 0) {
            network.push_back(std::unique_ptr<Layer>(new GBDropoutLayer(layerNodeCount.at(i), 0, inputLayerAF, groupSize, true, false)));
        }
        else if (i == layerNodeCount.size() - 1) {
            network.push_back(std::unique_ptr<Layer>(new GBDropoutLayer(layerNodeCount.at(i), layerNodeCount.at(i - 1), outputLayerAF, groupSize, false, true)));
        }
        else {
            network.push_back(std::unique_ptr<Layer>(new GBDropoutLayer(layerNodeCount.at(i), layerNodeCount.at(i - 1), hiddenLayerAF, groupSize, false, false)));
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
void testGBDropOutNetwork(std::vector<std::unique_ptr<Layer>>& gbDropOutNetwork, std::vector<std::vector<double>>& input,double scalingFactor, std::vector<double>& yTrue, std::string trainingFile, std::string predictFile) {
    std::ofstream gbDropOutLayerTrainingFile(trainingFile);
    std::ofstream gbDropOutLayerTestingFile(predictFile);
    std::cout << std::fixed << std::setprecision(7);
    gbDropOutLayerTrainingFile << std::fixed << std::setprecision(7);
    gbDropOutLayerTestingFile << std::fixed << std::setprecision(12);
    std::cout  << "----------Group Based Dropout Network----------" << std::endl;

    double yHat = 0.0;
    double loss = 0;
    double trainingLoss = 0.0;
    double validLoss = 0.0;

    int startOfTestIndex = (int)(input.size() * trainTestSplit);
    int startOfValidIndex = (int)(startOfTestIndex * trainValidSplit);

    for (int i = 0; i < epochs; i++) {
        trainingLoss = 0;

        //TRAINING
        for (int j = 0; j < startOfValidIndex; j++) {
            yHat = forwardProp(gbDropOutNetwork, input[j],scalingFactor)[0];
            loss = 2 * (yHat - (yTrue[j]));
            trainingLoss += std::pow((yHat - (yTrue[j])), 2);
            backProp(gbDropOutNetwork, loss);
            rollNodes(gbDropOutNetwork);
        }

        validLoss = 0;
        if(useAllNodesGBD){
            activateAllNodes(gbDropOutNetwork);
            if(scaleWeights){
                scaleAllWeights(gbDropOutNetwork);
            }
        }
        for (int j = startOfValidIndex; j < startOfTestIndex; j++) {
            if(!useAllNodesGBD){
                rollNodes(gbDropOutNetwork);
            }

            yHat = forwardProp(gbDropOutNetwork, input[j],1)[0];
            validLoss += std::pow((yHat - yTrue[j]), 2);
        }

        trainingLoss /= startOfValidIndex;
        validLoss /= (startOfTestIndex - startOfValidIndex);
        std::cout << "Epoch: " << i << "\t" << "Training Loss: " << trainingLoss << "\t" << "Valid Loss: " << validLoss << std::endl;
        gbDropOutLayerTrainingFile << trainingLoss << "," << validLoss << "\n";
    }

    loss = 0;

    std::vector<double> yHatMulti = std::vector<double>(3,0);
    double sum=0;
    double onceLoss = 0;
    yHat=0;
    //TESTING
    if(useAllNodesGBD){
        activateAllNodes(gbDropOutNetwork);
        if(scaleWeights){
            scaleAllWeights(gbDropOutNetwork);
        }
    }
    for (int j = startOfTestIndex; j < input.size(); j++) {
        for(int i=0;i<3;i++){
            if(!useAllNodesGBD){
                rollNodes(gbDropOutNetwork);
            }

            yHatMulti[i] = forwardProp(gbDropOutNetwork, input[j],1)[0];
            sum+=yHatMulti[i];
        }

        yHat = sum/3;
        loss += std::pow((yHat - yTrue[j]), 2);
        onceLoss += std::pow((yHatMulti[0] - yTrue[j]), 2);
        sum=0;
        gbDropOutLayerTestingFile << yHatMulti[0] << "," << yHatMulti[1] << "," << yHatMulti[2] << "," << yTrue[j] << "\n";
    }
    std::cout << loss << std::endl;
    std::cout << onceLoss << std::endl;
    std::cout << "GBDO = 3 Pass Test Loss: " << loss / (input.size() - startOfTestIndex) << "\t" << "1 Pass Test Loss: " << onceLoss / (input.size() - startOfTestIndex) << std::endl;
    gbDropOutLayerTestingFile << loss << "," << onceLoss << "," << "00" << "," << "00"<< "\n";

    gbDropOutLayerTrainingFile.close();
    gbDropOutLayerTestingFile.close();
}

//DROP OUT LAYER METHODS
std::vector<std::unique_ptr<Layer>> createDropOutNetwork(std::vector<int>& layerNodeCount, int dropOutRate) {
    std::vector<std::unique_ptr<Layer>> network;
    for (int i = 0; i < layerNodeCount.size(); i++) {
        if (i == 0) {
            network.push_back(std::unique_ptr<Layer>(new DropOutLayer(layerNodeCount.at(i), 0, inputLayerAF, dropOutRate, true, false)));
        }
        else if (i == layerNodeCount.size() - 1) {
            network.push_back(std::unique_ptr<Layer>(new DropOutLayer(layerNodeCount.at(i), layerNodeCount.at(i - 1), outputLayerAF, dropOutRate, false, true)));
        }
        else {
            network.push_back(std::unique_ptr<Layer>(new DropOutLayer(layerNodeCount.at(i), layerNodeCount.at(i - 1), hiddenLayerAF, dropOutRate, false, false)));
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
void testDropOutNetwork(std::vector<std::unique_ptr<Layer>>& dropOutNetwork, std::vector<std::vector<double>>& input,double scalingFactor, std::vector<double>& yTrue, std::string trainingFile, std::string predictFile) {
    std::ofstream dropOutLayerTrainingFile(trainingFile);
    std::ofstream dropOutLayerTestingFile(predictFile);
    std::cout << std::fixed << std::setprecision(7);
    dropOutLayerTrainingFile << std::fixed << std::setprecision(7);
    dropOutLayerTestingFile << std::fixed << std::setprecision(12);
    std::cout  << "----------Standard Dropout Network----------" << std::endl;

    double yHat = 0.0;
    double loss = 0;
    double trainingLoss = 0.0;
    double validLoss = 0.0;

    int startOfTestIndex = (int)(input.size() * trainTestSplit);
    int startOfValidIndex = (int)(startOfTestIndex * trainValidSplit);

    for (int i = 0; i < epochs; i++) {
        trainingLoss = 0;

        //TRAINING
        for (int j = 0; j < startOfValidIndex; j++) {
            yHat = forwardProp(dropOutNetwork, input[j],scalingFactor)[0];
            loss = 2 * (yHat - yTrue[j]);
            trainingLoss += std::pow((yHat - yTrue[j]), 2);
            backProp(dropOutNetwork, loss);
            rollNodes(dropOutNetwork);
        }

        validLoss = 0;
        if(useAllNodesDrop){
            activateAllNodes(dropOutNetwork);
        }
        for (int j = startOfValidIndex; j < startOfTestIndex; j++) {
            if(!useAllNodesDrop){
                rollNodes(dropOutNetwork);
            }

            yHat = forwardProp(dropOutNetwork, input[j],1)[0];
            validLoss += std::pow((yHat - yTrue[j]), 2);
        }
        trainingLoss /= startOfValidIndex;
        validLoss /= (startOfTestIndex - startOfValidIndex);
        std::cout << "Epoch: " << i << "\t" << "Training Loss: " << trainingLoss << "\t" << "Valid Loss: " << validLoss << "\t" << std::endl;
        dropOutLayerTrainingFile << trainingLoss << "," << validLoss << "\n";
    }

    loss = 0;
    double yHatOnce = 0;
    double onceLoss = 0;
    if(useAllNodesDrop){
        activateAllNodes(dropOutNetwork);
    }
    for (int j = startOfTestIndex; j < input.size(); j++) {
        if(!useAllNodesDrop){
            rollNodes(dropOutNetwork);
        }
        yHatOnce = forwardProp(dropOutNetwork, input[j],1)[0];
        yHat = yHatOnce;
        loss += std::pow((yHat - yTrue[j]), 2);
        onceLoss += std::pow((yHatOnce - yTrue[j]), 2);

        dropOutLayerTestingFile << yHat << "," << yHatOnce << "," << yTrue[j] << "\n";
    }
    std::cout << loss << std::endl;
    std::cout << onceLoss << std::endl;
    std::cout << "     DO = 3 Pass Test Loss: " << loss / (input.size() - startOfTestIndex) << "\t" << "   1 Pass Test Loss: " << onceLoss / (input.size() - startOfTestIndex) << std::endl;

    dropOutLayerTestingFile << loss << "," << onceLoss << "," << "00" << "\n";

    dropOutLayerTrainingFile.close();
    dropOutLayerTestingFile.close();
}


//SHARED METHODS
std::vector<double> forwardProp(std::vector< std::unique_ptr<Layer>>& network, std::vector<double>& data, double scalingFactor) {
    for (int layer = 0; layer < network.size(); layer++) {
        if (layer == 0)
            network.at(layer).get()->setOutput(data);
        else
            network.at(layer).get()->forwardPropagation(scalingFactor);
    }
    return network.at(network.size() - 1).get()->getOutput();
}
void backProp(std::vector<std::unique_ptr<Layer>>& network, double loss) {
    for (int layer = 1; layer < network.size(); layer++) {
        network.at(layer).get()->updateAllBiases(loss, learningRate);
        network.at(layer).get()->updateAllWeights(loss, learningRate);
    }
}
void rollNodes(std::vector<std::unique_ptr<Layer>>& network) {
    for (int i = 0; i < network.size(); i++) {
        network[i].get()->rollActiveLayers();
    }
}
void activateAllNodes(std::vector<std::unique_ptr<Layer>>& network) {
    for (int i = 0; i < network.size(); i++)
        network.at(i).get()->useAllNodes();
}
void scaleAllWeights(std::vector<std::unique_ptr<Layer>>& network){
    for(int i=1; i<network.size();i++){
        network.at(i)->scaleWeights();
    }
}
void shakeNetworkWeightsAndBiases(std::vector<std::unique_ptr<Layer>>& network, double delta) {
    for (int i = 1; i < network.size(); i++) {
        network[i].get()->shakeWeightsAndBiases(delta);
    }
}
void resetNetworkWeightsAndBiases(std::vector<std::unique_ptr<Layer>>& network) {
    for (int i = 1; i < network.size(); i++)
        network.at(i).get()->resetWeightsAndBias();
}



