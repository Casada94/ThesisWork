#include "FatLayer.h"
#include "DropOutLayer.h"
#include "MixedLayer.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include <random>

std::vector<std::unique_ptr<Layer>> createHybridNetwork(std::vector<int>& layerNodeCount, std::vector<int>& layerTypeDef, int layerDepth, int dropOutRate);

//FatLayer Methods
std::vector<std::unique_ptr<Layer>> createFatNetwork(std::vector<int>& layerNodeCount, int layerDepth);
void testFatNetwork(std::vector<std::unique_ptr<Layer>>& fatNetwork, int epochs, double learningRate, std::vector<std::vector<double>>& input, std::vector<double>& yTrue, std::string trainingFile, std::string predictFile);

//DropOutLayer Methods
std::vector<std::unique_ptr<Layer>> createDropOutNetwork(std::vector<int>& layerNodeCount, int dropOutRate);
void testDropOutNetwork(std::vector<std::unique_ptr<Layer>>& dropOutNetwork,int epochs, double learningRate, std::vector<std::vector<double>>& input, std::vector<double>& yTrue, std::string trainingFile, std::string predictFile);
void useAllNodes(std::vector<std::unique_ptr<Layer>>& network);

//MixedLayer Methods
std::vector<std::unique_ptr<Layer>> createMixedNetwork(std::vector<int>& layerNodeCount, int groupSize);
void testMixedNetwork(std::vector<std::unique_ptr<Layer>>& dropOutNetwork, int epochs, double learningRate, std::vector<std::vector<double>>& input, std::vector<double>& yTrue, std::string trainingFile, std::string predictFile);

//Shared
std::vector<double> forwardProp(std::vector< std::unique_ptr<Layer>>& network, std::vector<double>& data);
void backProp(std::vector<std::unique_ptr<Layer>>& network, double loss, double learningRate);
void resetNetworkWeightsAndBiases(std::vector<std::unique_ptr<Layer>>& network);
void rollNodes(std::vector<std::unique_ptr<Layer>>& dropOutNetwork);
void readDataSet(std::string filename, std::vector<std::vector<double>>& input);
void readDataSet(std::string filename, std::vector<double>& yTrue);
std::vector<std::string> splitString(const std::string& str, char delimiter);
void shuffleRows(std::vector<std::vector<double>>& data, std::vector<double>& yTrue, int size);
void shakeWeights(std::vector<std::unique_ptr<Layer>>& network,double lowShake,double highShake);


int main() {
    std::vector<int> fatLayerNodeCounts = { 8,10,5,1 };
    std::vector<int> mixLayerNodeCounts = {8,20,10,1};
    std::vector<int> dropOutLayerNodeCounts =	{ 8,20,10,1 };
//    std::vector<int> layerTypeDef = {0,1,0,0};
	int epochs = 250;
	double learningRate = .01;
	int dropOutPercent = 20;
    int groupSize = 2;
	int layerDepth = 2;

	std::vector<std::unique_ptr<Layer>> fatNetwork = createFatNetwork(fatLayerNodeCounts, layerDepth);
	std::vector<std::unique_ptr<Layer>> dropOutNetwork = createDropOutNetwork(dropOutLayerNodeCounts, dropOutPercent);
    std::vector<std::unique_ptr<Layer>> mixedNetwork = createMixedNetwork(dropOutLayerNodeCounts, groupSize);

//    std::vector<std::unique_ptr<Layer>> hybridNetwork = createHybridNetwork(fatLayerNodeCounts,layerTypeDef,layerDepth,dropOutPercent);

	std::vector<std::vector<double>> input(20640,std::vector<double>(8,0));
	std::vector<double> yTrue(20640,0);
	readDataSet("C:\\Users\\clayton\\Desktop\\Thesis\\ThesisWork\\input\\housingData.csv", input);
	readDataSet("C:\\Users\\clayton\\Desktop\\Thesis\\ThesisWork\\input\\housingDataTrueY.csv", yTrue);
	
	//ROUND 1
	shuffleRows(input, yTrue, yTrue.size());

	testFatNetwork(fatNetwork,epochs,learningRate,input,yTrue,
		"C:\\Users\\clayton\\Desktop\\Thesis\\ThesisWork\\output\\fatNode\\trainingLoss1.csv",
		"C:\\Users\\clayton\\Desktop\\Thesis\\ThesisWork\\output\\fatNode\\predictionOutput1.csv");
    testMixedNetwork(mixedNetwork, epochs, learningRate, input, yTrue,
                       "C:\\Users\\clayton\\Desktop\\Thesis\\ThesisWork\\output\\mixed\\trainingLoss1.csv",
                       "C:\\Users\\clayton\\Desktop\\Thesis\\ThesisWork\\output\\mixed\\predictionOutput1.csv");
//    testDropOutNetwork(dropOutNetwork, epochs, learningRate, input, yTrue,
//                       "C:\\Users\\clayton\\Desktop\\Thesis\\ThesisWork\\output\\dropOut\\trainingLoss1.csv",
//                       "C:\\Users\\clayton\\Desktop\\Thesis\\ThesisWork\\output\\dropOut\\predictionOutput1.csv");


}
//HYBRID NETWORK
std::vector<std::unique_ptr<Layer>> createHybridNetwork(std::vector<int>& layerNodeCount, std::vector<int>& layerTypeDef, int layerDepth, int dropOutRate){
    std::vector<std::unique_ptr<Layer>> network;
    for (int i = 0; i < layerNodeCount.size(); i++) {
        if(layerTypeDef[i]){
            if (i == 0) {
                network.push_back(std::unique_ptr<Layer>(new FatLayer(layerNodeCount.at(i), 0, layerDepth,2, true, false)));
            }
            else if (i == layerNodeCount.size() - 1) {
                network.push_back(std::unique_ptr<Layer>(new FatLayer(layerNodeCount.at(i), layerNodeCount.at(i-1), layerDepth, 3, false, true)));
            }
            else {
                network.push_back(std::unique_ptr<Layer>(new FatLayer(layerNodeCount.at(i), layerNodeCount.at(i - 1), layerDepth, 0, false, false)));
            }
        } else{
            if (i == 0) {
                network.push_back(std::unique_ptr<Layer>(new DropOutLayer(layerNodeCount.at(i), 0, 2, dropOutRate, true, false)));
            }
            else if (i == layerNodeCount.size() - 1) {
                network.push_back(std::unique_ptr<Layer>(new DropOutLayer(layerNodeCount.at(i), layerNodeCount.at(i - 1), 3, dropOutRate, false, true)));
            }
            else {
                network.push_back(std::unique_ptr<Layer>(new DropOutLayer(layerNodeCount.at(i), layerNodeCount.at(i - 1), 0, dropOutRate, false, false)));
            }
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


// FAT LAYER METHODS
std::vector<std::unique_ptr<Layer>> createFatNetwork(std::vector<int>& layerNodeCount, int layerDepth) {
	std::vector<std::unique_ptr<Layer>> network;
	for (int i = 0; i < layerNodeCount.size(); i++) {
		if (i == 0) {
			network.push_back(std::unique_ptr<Layer>(new FatLayer(layerNodeCount.at(i), 0, layerDepth,2, true, false)));
		}
		else if (i == layerNodeCount.size() - 1) {
			network.push_back(std::unique_ptr<Layer>(new FatLayer(layerNodeCount.at(i), layerNodeCount.at(i-1), layerDepth, 3, false, true)));
		}
		else {
			network.push_back(std::unique_ptr<Layer>(new FatLayer(layerNodeCount.at(i), layerNodeCount.at(i - 1), layerDepth, 0, false, false)));
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
void testFatNetwork(std::vector<std::unique_ptr<Layer>>& fatNetwork, int epochs, double learningRate, std::vector<std::vector<double>>& input, std::vector<double>& yTrue,std::string trainingFile,std::string predictFile) {
	std::ofstream fatLayerTrainingFile(trainingFile);
	std::ofstream fatLayerTestingFile(predictFile);

	double yHat = 0.0;
	double loss = 0;
	double trainingLoss = 0.0;
	double validLoss = 0.0;

    int startOfTestIndex = (int)(input.size() * .75);
	int startOfValidIndex = (int)(startOfTestIndex * .5);

    std::cout << std::fixed << std::setprecision(7);
    fatLayerTrainingFile << std::fixed << std::setprecision(7);
    fatLayerTestingFile << std::fixed << std::setprecision(12);

	for (int i = 0; i < epochs; i++) {
		trainingLoss = 0;
		for (int j = 0; j < startOfValidIndex; j++) {
			yHat = forwardProp(fatNetwork, input[j])[0];
			loss = 2 * (yHat - yTrue[j]);
			trainingLoss += std::pow((yHat - yTrue[j]), 2);
			backProp(fatNetwork, loss, learningRate);
			rollNodes(fatNetwork);
		}
		validLoss = 0;
		for (int j = startOfValidIndex; j < startOfTestIndex; j++) {
			yHat = forwardProp(fatNetwork, input[j])[0];
			validLoss += std::pow((yHat - yTrue[j]), 2);
			rollNodes(fatNetwork);
		}
		trainingLoss /= startOfValidIndex;
		validLoss /= (startOfTestIndex - startOfValidIndex);
		std::cout  << "Epoch: " << i << "\t" << "Training Loss: " << trainingLoss<< "\t" << "Valid Loss: " << validLoss << std::endl;
		fatLayerTrainingFile  << trainingLoss << "," << validLoss  << "\n";
	}

	loss = 0;
	//double yHatOnce = 0;
    double first =0;
    double second =0;
    double third=0;
    double onceLoss = 0;
    yHat=0;
	for (int j = startOfTestIndex; j < input.size(); j++) {
		rollNodes(fatNetwork);
		first = forwardProp(fatNetwork, input[j])[0];

        rollNodes(fatNetwork);
        second = forwardProp(fatNetwork, input[j])[0];

        rollNodes(fatNetwork);
        third = forwardProp(fatNetwork, input[j])[0];

        yHat = (first+second+third)/3;
		loss += std::pow((yHat - yTrue[j]), 2);
		onceLoss += std::pow((first - yTrue[j]), 2);

//        fatLayerTestingFile << yHat << "," << yHatOnce << "," << yTrue[j] << "\n";
        fatLayerTestingFile << first << "," << second << "," << third << "," << yTrue[j] << "\n";

    }
	
	std::cout << loss << std::endl;
	std::cout << onceLoss << std::endl;
	std::cout << std::fixed << std::setprecision(7) << "3 Pass Test Loss: " << loss / (input.size() - startOfTestIndex) << "\t" << "1 Pass Test Loss: " << onceLoss / (input.size() - startOfTestIndex) << std::endl;
//	fatLayerTestingFile << loss << "," << onceLoss << "," << "00" << "\n";
    fatLayerTestingFile << loss << "," << onceLoss << "," << "00" << "," << "00" << "\n";

	fatLayerTrainingFile.close();
	fatLayerTestingFile.close();
}

//DROP OUT LAYER METHODS
std::vector<std::unique_ptr<Layer>> createDropOutNetwork(std::vector<int>& layerNodeCount, int dropOutRate) {
    std::vector<std::unique_ptr<Layer>> network;
    for (int i = 0; i < layerNodeCount.size(); i++) {
        if (i == 0) {
            network.push_back(std::unique_ptr<Layer>(new DropOutLayer(layerNodeCount.at(i), 0, 2, dropOutRate, true, false)));
        }
        else if (i == layerNodeCount.size() - 1) {
            network.push_back(std::unique_ptr<Layer>(new DropOutLayer(layerNodeCount.at(i), layerNodeCount.at(i - 1), 3, dropOutRate, false, true)));
        }
        else {
            network.push_back(std::unique_ptr<Layer>(new DropOutLayer(layerNodeCount.at(i), layerNodeCount.at(i - 1), 0, dropOutRate, false, false)));
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
void testDropOutNetwork(std::vector<std::unique_ptr<Layer>>& dropOutNetwork, int epochs, double learningRate, std::vector<std::vector<double>>& input, std::vector<double>& yTrue, std::string trainingFile, std::string predictFile) {
    std::ofstream dropOutLayerTrainingFile(trainingFile);
    std::ofstream dropOutLayerTestingFile(predictFile);

    double yHat = 0.0;
    double loss = 0;
    double trainingLoss = 0.0;
    double validLoss = 0.0;

    int startOfTestIndex = (int)(input.size() * .75);
    int startOfValidIndex = (int)(startOfTestIndex * .5);

    std::cout << std::fixed << std::setprecision(7);
    dropOutLayerTrainingFile << std::fixed << std::setprecision(7);
    dropOutLayerTestingFile << std::fixed << std::setprecision(12);
    for (int i = 0; i < epochs; i++) {
        rollNodes(dropOutNetwork);
        trainingLoss = 0;
        for (int j = 0; j < startOfValidIndex; j++) {
            yHat = forwardProp(dropOutNetwork, input[j])[0];
            loss = 2 * (yHat - yTrue[j]);
            trainingLoss += std::pow((yHat - yTrue[j]), 2);
            backProp(dropOutNetwork, loss, learningRate);
            rollNodes(dropOutNetwork);
        }
        validLoss = 0;
        useAllNodes(dropOutNetwork);
        for (int j = startOfValidIndex; j < startOfTestIndex; j++) {
            yHat = forwardProp(dropOutNetwork, input[j])[0];
            validLoss += std::pow((yHat - yTrue[j]), 2);
        }
        std::cout << "Epoch: " << i << "\t" << "Training Loss: " << trainingLoss / startOfValidIndex << "\t" << "Valid Loss: " << validLoss / (startOfTestIndex - startOfValidIndex) << "\t" << std::endl;
        dropOutLayerTrainingFile << trainingLoss / startOfValidIndex << "," << validLoss / (startOfTestIndex - startOfValidIndex) << "\n";
    }

    loss = 0;
    double yHatOnce = 0;
    double onceLoss = 0;
    for (int j = startOfTestIndex; j < input.size(); j++) {
        yHatOnce = forwardProp(dropOutNetwork, input[j])[0];
        yHat = yHatOnce;
        loss += std::pow((yHat - yTrue[j]), 2);
        onceLoss += std::pow((yHatOnce - yTrue[j]), 2);

        dropOutLayerTestingFile << yHat << "," << yHatOnce << "," << yTrue[j] << "\n";
    }
    std::cout << loss << std::endl;
    std::cout << onceLoss << std::endl;
    std::cout << "3 Pass Test Loss: " << loss / (input.size() - startOfTestIndex) << "\t" << "1 Pass Test Loss: " << onceLoss / (input.size() - startOfTestIndex) << std::endl;

    dropOutLayerTestingFile << loss << "," << onceLoss << "," << "00" << "\n";

    dropOutLayerTrainingFile.close();
    dropOutLayerTestingFile.close();
}
void useAllNodes(std::vector<std::unique_ptr<Layer>>& dropOutNetwork) {
    for (int i = 1; i < dropOutNetwork.size(); i++)
        dropOutNetwork.at(i).get()->useAllNodes();
}

//MIXED NETWORK
std::vector<std::unique_ptr<Layer>> createMixedNetwork(std::vector<int>& layerNodeCount, int groupSize) {
    std::vector<std::unique_ptr<Layer>> network;
    for (int i = 0; i < layerNodeCount.size(); i++) {
        if (i == 0) {
            network.push_back(std::unique_ptr<Layer>(new MixedLayer(layerNodeCount.at(i), 0, 2, groupSize, true, false)));
        }
        else if (i == layerNodeCount.size() - 1) {
            network.push_back(std::unique_ptr<Layer>(new MixedLayer(layerNodeCount.at(i), layerNodeCount.at(i - 1), 3, groupSize, false, true)));
        }
        else {
            network.push_back(std::unique_ptr<Layer>(new MixedLayer(layerNodeCount.at(i), layerNodeCount.at(i - 1), 0, groupSize, false, false)));
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
void testMixedNetwork(std::vector<std::unique_ptr<Layer>>& mixedNetwork, int epochs, double learningRate, std::vector<std::vector<double>>& input, std::vector<double>& yTrue, std::string trainingFile, std::string predictFile) {
    std::ofstream mixedLayerTrainingFile(trainingFile);
    std::ofstream mixedLayerTestingFile(predictFile);

    double yHat = 0.0;
    double loss = 0;
    double trainingLoss = 0.0;
    double validLoss = 0.0;

    int startOfTestIndex = (int)(input.size() * .75);
    int startOfValidIndex = (int)(startOfTestIndex * .5);

    std::cout << std::fixed << std::setprecision(7);
    mixedLayerTrainingFile << std::fixed << std::setprecision(7);
    mixedLayerTestingFile << std::fixed << std::setprecision(12);

    for (int i = 0; i < epochs; i++) {
        rollNodes(mixedNetwork);
        trainingLoss = 0;
        for (int j = 0; j < startOfValidIndex; j++) {
            yHat = forwardProp(mixedNetwork, input[j])[0];
            loss = 2 * (yHat - yTrue[j]);
            trainingLoss += std::pow((yHat - yTrue[j]), 2);
            backProp(mixedNetwork, loss, learningRate);
            rollNodes(mixedNetwork);
        }
        validLoss = 0;
        for (int j = startOfValidIndex; j < startOfTestIndex; j++) {
            yHat = forwardProp(mixedNetwork, input[j])[0];
            validLoss += std::pow((yHat - yTrue[j]), 2);
            rollNodes(mixedNetwork);
        }
        trainingLoss /= startOfValidIndex;
        validLoss /= (startOfTestIndex - startOfValidIndex);
        std::cout << "Epoch: " << i << "\t" << "Training Loss: " << trainingLoss << "\t" << "Valid Loss: " << validLoss << std::endl;
        mixedLayerTrainingFile << trainingLoss << "," << validLoss << "\n";
    }

    loss = 0;

    std::vector<double> yHatMult = std::vector<double>(3,0);
    double sum=0;
    double onceLoss = 0;
    yHat=0;
    for (int j = startOfTestIndex; j < input.size(); j++) {
        for(int i=0;i<3;i++){
            rollNodes(mixedNetwork);
            yHatMult[i] = forwardProp(mixedNetwork, input[j])[0];
            sum+=yHatMult[i];
        }

        yHat = sum/3;
        loss += std::pow((yHat - yTrue[j]), 2);
        onceLoss += std::pow((yHatMult[0] - yTrue[j]), 2);
        sum=0;
        mixedLayerTestingFile << yHatMult[0] << "," << yHatMult[1] << "," << yHatMult[2] << "," << yTrue[j] << "\n";

    }
    std::cout << loss << std::endl;
    std::cout << onceLoss << std::endl;
    std::cout << "3 Pass Test Loss: " << loss / (input.size() - startOfTestIndex) << "\t" << "1 Pass Test Loss: " << onceLoss / (input.size() - startOfTestIndex) << std::endl;

    mixedLayerTestingFile << loss << "," << onceLoss << "," << "00" << "," << "00"<< "\n";

    mixedLayerTrainingFile.close();
    mixedLayerTestingFile.close();
}

//SHARED METHODS
void readDataSet(std::string filename, std::vector<std::vector<double>>& input) {
	std::ifstream file(filename);  // Open the file

	if (!file.is_open()) {
		std::cerr << "Could not open the file: " << filename << std::endl;
		return;
	}

	std::string line;
	int count = 0;
	while (std::getline(file, line)) {
		std::vector<std::string> tokens = splitString(line, ',');
		for (int i = 0; i < tokens.size(); i++) {
			//input[count][i] = std::stod(tokens[i]);
			switch (i) {
				case 0:
					input[count][i] = (std::stod(tokens[i]) - 3.870671003) / 1.899821718;
					break;
				case 1:
					input[count][i] = (std::stod(tokens[i]) - 28.63948643) / 12.58555761;
					break;
				case 2:
					input[count][i] = (std::stod(tokens[i]) - 5.428999742) / 2.474173139;
					break;
				case 3:
					input[count][i] = (std::stod(tokens[i]) - 1.09667515) / 0.473910857;
					break;
				case 4:
					input[count][i] = (std::stod(tokens[i]) - 1425.476744) / 1132.462122;
					break;
				case 5:
					input[count][i] = (std::stod(tokens[i]) - 3.070655159) / 10.38604956;
					break;
				case 6:
					input[count][i] = (std::stod(tokens[i]) - 35.63186143) / 2.135952397;
					break;
				case 7:
					input[count][i] = (std::stod(tokens[i]) + 119.5697045) / 2.003531724;
					break;
			}
		}
		count++;
	}
	file.close();  // Close the file
}
void readDataSet(std::string filename, std::vector<double>& yTrue) {
	std::ifstream file(filename);  // Open the file

	if (!file.is_open()) {
		std::cerr << "Could not open the file: " << filename << std::endl;
		return;
	}

	std::string line;
	int count = 0;
	while (std::getline(file, line)) {
		yTrue[count++] = std::stod(line);
	}
	file.close();  // Close the file
}
std::vector<std::string> splitString(const std::string& str, char delimiter) {
	std::vector<std::string> tokens;
	std::istringstream ss(str);
	std::string token;

	while (std::getline(ss, token, delimiter)) {
		tokens.push_back(token);
	}
	return tokens;
}
void shuffleRows(std::vector<std::vector<double>>& data, std::vector<double>&yTrue, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distribution(0, size-1);

	int first = 0;
	for(int j=0;j<3;j++){
        for (int i = 0; i < size; i++) {
            first = distribution(gen);
            std::swap(data[i], data[first]);
            std::swap(yTrue[i], yTrue[first]);
        }
    }
}
std::vector<double> forwardProp(std::vector< std::unique_ptr<Layer>>& network, std::vector<double>& data) {
	for (int layer = 0; layer < network.size(); layer++) {
		if (layer == 0)
			network.at(layer).get()->setOutput(data);
		else
			network.at(layer).get()->forwardPropagation();
	}
	return network.at(network.size() - 1).get()->getOutput();
}
void backProp(std::vector<std::unique_ptr<Layer>>& network, double loss, double learningRate) {
	for (int layer = 1; layer < network.size(); layer++) {
		network.at(layer).get()->updateAllBiases(loss, learningRate);
		network.at(layer).get()->updateAllWeights(loss, learningRate);
	}
}
void rollNodes(std::vector<std::unique_ptr<Layer>>& network) {
	for (int i = 1; i < network.size(); i++) {
		network[i].get()->rollActiveLayers();
	}
}
void shakeWeights(std::vector<std::unique_ptr<Layer>>& network, double lowShake, double highShake) {
	std::cout << "shaking weights" << std::endl;
	for (int i = 1; i < network.size(); i++) {
		network[i].get()->shakeWeights(lowShake,highShake);
	}
}
void resetNetworkWeightsAndBiases(std::vector<std::unique_ptr<Layer>>& dropOutNetwork) {
	for (int i = 1; i < dropOutNetwork.size(); i++)
		dropOutNetwork.at(i).get()->resetWeightsAndBias();
}


