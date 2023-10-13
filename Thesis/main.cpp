#include "FatLayer.h"
#include "DropOutLayer.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <random>
#include <vector>
#include <cmath>
#include <algorithm>

//FatLayer Methods
std::vector<std::unique_ptr<Layer>> createFatNetwork(std::vector<int>& layerNodeCount, int layerDepth);
void testFatNetwork(std::vector<std::unique_ptr<Layer>>& fatNetwork, int epochs, double learningRate, std::vector<std::vector<double>>& input, std::vector<double>& yTrue, std::string trainingFile, std::string predictFile);

//DropOutLayer Methods
std::vector<std::unique_ptr<Layer>> createDropOutNetwork(std::vector<int>& layerNodeCount, int dropOutRate);
//std::vector<double> forwardProp(std::vector<std::unique_ptr<Layer>>& dropOutNetwork, double data);
void rollNodes(std::vector<std::unique_ptr<Layer>>& dropOutNetwork);
void testDropOutNetwork(std::vector<std::unique_ptr<Layer>>& dropOutNetwork,int epochs, double learningRate, std::vector<std::vector<double>>& input, std::vector<double>& yTrue, std::string trainingFile, std::string predictFile);

//Shared
std::vector<double> forwardProp(std::vector< std::unique_ptr<Layer>>& network, std::vector<double>& data);
void backProp(std::vector<std::unique_ptr<Layer>>& network, double loss, double learningRate);
void resetNetworkWeightsAndBiases(std::vector<std::unique_ptr<Layer>>& network);
void useAllNodes(std::vector<std::unique_ptr<Layer>>& network);
void readDataSet(std::string filename, std::vector<std::vector<double>>& input);
void readDataSet(std::string filename, std::vector<double>& yTrue);
std::vector<std::string> splitString(const std::string& str, char delimiter);
void shuffleRows(std::vector<std::vector<double>>& data, std::vector<double>& yTrue, int size);
void shakeWeights(std::vector<std::unique_ptr<Layer>>& network,double lowShake,double highShake);

int main() {
	std::vector<int> fatLayerNodeCounts = { 8,10,50,1 };
	std::vector<int> dropOutLayerNodeCounts =	{ 8,20,10,1 };
	int epochs = 300;
	double learningRate = .01; 
	int dropOutPercent = 10;
	int layerDepth = 2;

	std::vector<std::unique_ptr<Layer>> fatNetwork = createFatNetwork(fatLayerNodeCounts, layerDepth);
	std::vector<std::unique_ptr<Layer>> dropOutNetwork = createDropOutNetwork(dropOutLayerNodeCounts, dropOutPercent);

	std::vector<std::vector<double>> input(20640,std::vector<double>(8,0));
	std::vector<double> yTrue(20640,0);
	readDataSet("C:/Users/clayton/Desktop/Thesis/ThesisWork/input/housingData.csv", input);
	readDataSet("C:/Users/clayton/Desktop/Thesis/ThesisWork/input/housingDataTrueY.csv", yTrue);
	
	//ROUND 1
	shuffleRows(input, yTrue, yTrue.size());
	testFatNetwork(fatNetwork,epochs,learningRate,input,yTrue,
		"C:/Users/metal/Desktop/Thesis/output/fatNode/trainingLoss1.csv",
		"C:/Users/metal/Desktop/Thesis/output/fatNode/predictionOutput1.csv");
	//testDropOutNetwork(dropOutNetwork, epochs, learningRate, input, yTrue,
	//	"C:/Users/metal/Desktop/Thesis/output/dropOutNode/trainingLoss1.csv",
	//	"C:/Users/metal/Desktop/Thesis/output/dropOutNode/predictionOutput1.csv");

	
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
	int startOfValidIndex = (int)(startOfTestIndex * .75);
	double highShake = 1.1;
	double lowShake = .9;
	std::vector<bool> valGreaterTrain = { false,false,false,false,false };

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
		valGreaterTrain[i % 5] = validLoss > trainingLoss;
		std::cout << std::fixed << std::setprecision(7) << "Epoch: " << i << "\t" << "Training Loss: " << trainingLoss<< "\t" << "Valid Loss: " << validLoss << "\t" << std::endl;
		fatLayerTrainingFile << std::fixed << std::setprecision(7)  << trainingLoss << "," << validLoss  << "\n";
	
		if (valGreaterTrain[0] && valGreaterTrain[1] && valGreaterTrain[2] && valGreaterTrain[3] && valGreaterTrain[4] && (epochs-i>15)) {
			shakeWeights(fatNetwork,lowShake,highShake);
			valGreaterTrain[0]=false, valGreaterTrain[1]=false, valGreaterTrain[2]=false, valGreaterTrain[3]=false, valGreaterTrain[4] = false;
			lowShake *= lowShake;
			highShake *= highShake;
		}
	}

	loss = 0;
	double yHatOnce = 0;
	double onceLoss = 0;
	for (int j = startOfTestIndex; j < input.size(); j++) {
		rollNodes(fatNetwork);
		yHatOnce = forwardProp(fatNetwork, input[j])[0];
		yHat = yHatOnce;
		for (int k = 0; k < 2; k++) {
			rollNodes(fatNetwork);
			yHat += forwardProp(fatNetwork, input[j])[0];
		}
		yHat /= 3;
		loss += std::pow((yHat - yTrue[j]), 2);
		onceLoss += std::pow((yHatOnce - yTrue[j]), 2);

		fatLayerTestingFile << yHat << "," << yHatOnce << "," << yTrue[j] << "\n";
	}
	
	std::cout << loss << std::endl;
	std::cout << onceLoss << std::endl;
	std::cout << std::fixed << std::setprecision(7) << "3 Pass Test Loss: " << loss / (input.size() - startOfTestIndex) << "\t" << "1 Pass Test Loss: " << onceLoss / (input.size() - startOfTestIndex) << std::endl;
	fatLayerTestingFile << loss << "," << onceLoss << "," << "00" << "\n";

	fatLayerTrainingFile.close();
	fatLayerTestingFile.close();
}
void useAllNodes(std::vector<std::unique_ptr<Layer>>& dropOutNetwork) {
	for (int i = 1; i < dropOutNetwork.size(); i++)
		dropOutNetwork.at(i).get()->useAllNodes();
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
	double traingingLoss = 0.0;
	double validLoss = 0.0;
	
	int startOfTestIndex = (int)(input.size() * .75);
	int startOfValidIndex = (int)(startOfTestIndex * .75);

	for (int i = 0; i < epochs; i++) {
		rollNodes(dropOutNetwork);
		traingingLoss = 0;
		for (int j = 0; j < startOfValidIndex; j++) {
			yHat = forwardProp(dropOutNetwork, input[j])[0];
			loss = 2 * (yHat - yTrue[j]);
			traingingLoss += std::pow((yHat - yTrue[j]), 2);
			backProp(dropOutNetwork, loss, learningRate);
			rollNodes(dropOutNetwork);
		}
		validLoss = 0;
		useAllNodes(dropOutNetwork);
		for (int j = startOfValidIndex; j < startOfTestIndex; j++) {
			yHat = forwardProp(dropOutNetwork, input[j])[0];
			validLoss += std::pow((yHat - yTrue[j]), 2);
		}
		std::cout << std::fixed << std::setprecision(7) << "Epoch: " << i << "\t" << "Training Loss: " << traingingLoss / startOfValidIndex << "\t" << "Valid Loss: " << validLoss / (startOfTestIndex - startOfValidIndex) << "\t" << std::endl;
		dropOutLayerTrainingFile << std::fixed << std::setprecision(7) << traingingLoss / startOfValidIndex << "," << validLoss / (startOfTestIndex - startOfValidIndex) << "\n";
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
	std::cout << std::fixed << std::setprecision(7) << "3 Pass Test Loss: " << loss / (input.size() - startOfTestIndex) << "\t" << "1 Pass Test Loss: " << onceLoss / (input.size() - startOfTestIndex) << std::endl;

	dropOutLayerTestingFile << loss << "," << onceLoss << "," << "00" << "\n";

	dropOutLayerTrainingFile.close();
	dropOutLayerTestingFile.close();
}
void resetNetworkWeightsAndBiases(std::vector<std::unique_ptr<Layer>>& dropOutNetwork) {
	for (int i = 1; i < dropOutNetwork.size(); i++)
		dropOutNetwork.at(i).get()->resetWeightsAndBias();
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
	std::srand(static_cast<unsigned>(std::time(nullptr)));
	int first = 0;
	int second =0;
	for (int i = 0; i < size; i++) {
		first = std::rand() % (size);
		second = std::rand() % (size);
		std::swap(data[first], data[second]);
		std::swap(yTrue[first], yTrue[second]);
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
		network.at(i).get()->rollActiveLayers();
	}
}
void shakeWeights(std::vector<std::unique_ptr<Layer>>& network, double lowShake, double highShake) {
	std::cout << "shaking weights" << std::endl;
	for (int i = 1; i < network.size(); i++) {
		network[i].get()->shakeWeights(lowShake,highShake);
	}
}