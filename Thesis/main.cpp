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
std::vector<FatLayer> createFatNetwork(std::vector<int>& layerNodeCount, int layerDepth);
std::vector<double> forwardProp(std::vector<FatLayer>& network, std::vector<double>& data);
std::vector<double> forwardProp(std::vector<FatLayer>& network, double data);
void backProp(std::vector<FatLayer>& network, double loss, double learningRate);
void rollNodes(std::vector<FatLayer>& network);
void testFatNetwork(std::vector<FatLayer>& fatNetwork, int epochs, double learningRate, std::vector<std::vector<double>>& input, std::vector<double>& yTrue, std::string trainingFile, std::string predictFile);
void resetNetworkWeightsAndBiases(std::vector<FatLayer>& fatNetwork);

//DropOutLayer Methods
std::vector<DropOutLayer> createDropOutNetwork(std::vector<int>& layerNodeCount, int dropOutRate);
std::vector<double> forwardProp(std::vector<DropOutLayer>& dropOutNetwork, std::vector<double>& data);
std::vector<double> forwardProp(std::vector<DropOutLayer>& dropOutNetwork, double data);
void backProp(std::vector<DropOutLayer>& dropOutNetwork, double loss, double learningRate);
void rollNodes(std::vector<DropOutLayer>& dropOutNetwork);
void testDropOutNetwork(std::vector<DropOutLayer>& dropOutNetwork,int epochs, double learningRate, std::vector<std::vector<double>>& input, std::vector<double>& yTrue, std::string trainingFile, std::string predictFile);
void resetNetworkWeightsAndBiases(std::vector<DropOutLayer>& dropOutNetwork);

//Shared
void readDataSet(std::string filename, std::vector<std::vector<double>>& input);
void readDataSet(std::string filename, std::vector<double>& yTrue);
std::vector<std::string> splitString(const std::string& str, char delimiter);
void shuffleRows(std::vector<std::vector<double>>& data, std::vector<double>& yTrue, int size);

void main() {
	std::vector<int> fatLayerNodeCounts = { 8,10,5,1 };
	std::vector<int> fatLayerNodeCounts2 = { 8,20,10,1 };
	//std::vector<int> dropOutLayerNodeCounts =	{ 8,20,10,1 };
	int epochs = 500;
	double learningRate = .01; 
	int dropOutPercent = 75;
	int layerDepth = 5;

	std::vector<FatLayer> fatNetwork = createFatNetwork(fatLayerNodeCounts, layerDepth);
	std::vector<FatLayer> fatNetwork2 = createFatNetwork(fatLayerNodeCounts2, layerDepth);
	//std::vector<DropOutLayer> dropOutNetwork = createDropOutNetwork(dropOutLayerNodeCounts, dropOutPercent);

	std::vector<std::vector<double>> input(20640,std::vector<double>(8,0));
	std::vector<double> yTrue(20640,0);
	readDataSet("C:/Users/metal/Desktop/Thesis/input/housingData.csv", input);
	readDataSet("C:/Users/metal/Desktop/Thesis/input/housingDataTrueY.csv", yTrue);
	
	//ROUND 1
	shuffleRows(input, yTrue, yTrue.size());
	testFatNetwork(fatNetwork, epochs, learningRate, input, yTrue,
		"C:/Users/metal/Desktop/Thesis/output/fatNode/trainingLoss1.csv",
		"C:/Users/metal/Desktop/Thesis/output/fatNode/predictionOutput1.csv");
	testFatNetwork(fatNetwork2, epochs, learningRate, input, yTrue,
		"C:/Users/metal/Desktop/Thesis/output/fatNode/trainingLoss2.csv",
		"C:/Users/metal/Desktop/Thesis/output/fatNode/predictionOutput2.csv");

	//ROUND 2
	shuffleRows(input, yTrue, yTrue.size());
	resetNetworkWeightsAndBiases(fatNetwork);
	resetNetworkWeightsAndBiases(fatNetwork2);
	testFatNetwork(fatNetwork, epochs, learningRate, input, yTrue,
		"C:/Users/metal/Desktop/Thesis/output/fatNode/trainingLoss3.csv",
		"C:/Users/metal/Desktop/Thesis/output/fatNode/predictionOutput3.csv");
	testFatNetwork(fatNetwork2, epochs, learningRate, input, yTrue,
		"C:/Users/metal/Desktop/Thesis/output/fatNode/trainingLoss4.csv",
		"C:/Users/metal/Desktop/Thesis/output/fatNode/predictionOutput4.csv");
	
	//ROUND 3
	shuffleRows(input, yTrue, yTrue.size());
	resetNetworkWeightsAndBiases(fatNetwork);
	resetNetworkWeightsAndBiases(fatNetwork2);
	testFatNetwork(fatNetwork, epochs, learningRate, input, yTrue,
		"C:/Users/metal/Desktop/Thesis/output/fatNode/trainingLoss5.csv",
		"C:/Users/metal/Desktop/Thesis/output/fatNode/predictionOutput5.csv");
	testFatNetwork(fatNetwork2, epochs, learningRate, input, yTrue,
		"C:/Users/metal/Desktop/Thesis/output/fatNode/trainingLoss6.csv",
		"C:/Users/metal/Desktop/Thesis/output/fatNode/predictionOutput6.csv");

	//testDropOutNetwork(dropOutNetwork, epochs, learningRate, input, yTrue,
	//	"C:/Users/metal/Desktop/Thesis/output/dropOutNode/trainingLoss1.csv",
	//	"C:/Users/metal/Desktop/Thesis/output/dropOutNode/predictionOutput1.csv");
	//testDropOutNetwork(dropOutNetwork, epochs, learningRate, input, yTrue,
	//	"C:/Users/metal/Desktop/Thesis/output/dropOutNode/trainingLoss2.csv",
	//	"C:/Users/metal/Desktop/Thesis/output/dropOutNode/predictionOutput2.csv");
	//shuffleRows(input, yTrue, yTrue.size());

	
 }

// FAT LAYER METHODS
std::vector<FatLayer> createFatNetwork(std::vector<int>& layerNodeCount, int layerDepth) {
	std::vector<FatLayer> network;
	for (int i = 0; i < layerNodeCount.size(); i++) {
		if (i == 0) {
			network.push_back(FatLayer(layerNodeCount.at(i), 0, layerDepth,2, true, false));
		}
		else if (i == layerNodeCount.size() - 1) {
			network.push_back(FatLayer(layerNodeCount.at(i), layerNodeCount.at(i-1), layerDepth, 3, false, true));
		}
		else {
			network.push_back(FatLayer(layerNodeCount.at(i), layerNodeCount.at(i - 1), layerDepth, 0, false, false));
		}
	}
	for (int i = 0; i < network.size(); i++) {
		if (i == 0) {
			network.at(i).setNextLayer(&network.at(i+1));
		}
		else if (i == network.size() - 1) {
			network.at(i).setPreviousLayer(&network.at(i - 1));
		}
		else {
			network.at(i).setNextLayer(&network.at(i + 1));
			network.at(i).setPreviousLayer(&network.at(i - 1));
		}
	}
	return network;
}
std::vector<double> forwardProp(std::vector<FatLayer>& network, std::vector<double>& data) {
	for (int layer = 0; layer < network.size(); layer++) {
		if (layer == 0)
			network.at(layer).setOutput(data);
		else
			network.at(layer).forwardPropagation();
	}
	return network.at(network.size() - 1).getOutput();
}
std::vector<double> forwardProp(std::vector<FatLayer>& network, double data) {
	for (int layer = 0; layer < network.size(); layer++) {
		if (layer == 0)
			network.at(layer).setOutput(data);
		else
			network.at(layer).forwardPropagation();
	}
	return network.at(network.size() - 1).getOutput();
}
void backProp(std::vector<FatLayer>& network, double loss, double learningRate) {
	for (int layer = 1; layer < network.size(); layer++) {
		network.at(layer).updateAllBiases(loss, learningRate);
		network.at(layer).updateAllWeights(loss, learningRate);
	}
}
void rollNodes(std::vector<FatLayer>& network) {
	for (int i = 0; i < network.size(); i++) {
		network.at(i).rollActiveLayers();
	}
}
void testFatNetwork(std::vector<FatLayer>& fatNetwork, int epochs, double learningRate, std::vector<std::vector<double>>& input, std::vector<double>& yTrue,std::string trainingFile,std::string predictFile) {
	std::ofstream fatLayerTrainingFile(trainingFile);
	std::ofstream fatLayerTestingFile(predictFile);

	double yHat = 0.0;
	double loss = 0;
	double traingingLoss = 0.0;
	double validLoss = 0.0;
	int startOfTestIndex = (int)(input.size() * .75);
	int startOfValidIndex = (int)(startOfTestIndex * .75);

	for (int i = 0; i < epochs; i++) {
		traingingLoss = 0;
		for (int j = 0; j < startOfValidIndex; j++) {
			yHat = forwardProp(fatNetwork, input[j])[0];
			loss = 2 * (yHat - yTrue[j]);
			traingingLoss += std::pow((yHat - yTrue[j]), 2);
			backProp(fatNetwork, loss, learningRate);
			rollNodes(fatNetwork);
		}
		validLoss = 0;
		for (int j = startOfValidIndex; j < startOfTestIndex; j++) {
			yHat = forwardProp(fatNetwork, input[j])[0];
			validLoss += std::pow((yHat - yTrue[j]), 2);
			rollNodes(fatNetwork);
		}
		std::cout << std::fixed << std::setprecision(7) << "Epoch: " << i << "\t" << "Training Loss: " << traingingLoss / startOfValidIndex << "\t" << "Valid Loss: " << validLoss / (startOfTestIndex - startOfValidIndex) << "\t" << std::endl;
		fatLayerTrainingFile << std::fixed << std::setprecision(7)  << traingingLoss / startOfValidIndex << "," << validLoss / (startOfTestIndex - startOfValidIndex) << "\n";
	
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
void resetNetworkWeightsAndBiases(std::vector<FatLayer>& fatNetwork) {
	for (int i = 1; i < fatNetwork.size(); i++)
		fatNetwork.at(i).resetWeightsAndBias();
}

//DROP OUT LAYER METHODS
std::vector<DropOutLayer> createDropOutNetwork(std::vector<int>& layerNodeCount, int dropOutRate) {
	std::vector<DropOutLayer> network;
	for (int i = 0; i < layerNodeCount.size(); i++) {
		if (i == 0) {
			network.push_back(DropOutLayer(layerNodeCount.at(i), 0, 2, dropOutRate, true, false));
		}
		else if (i == layerNodeCount.size() - 1) {
			network.push_back(DropOutLayer(layerNodeCount.at(i), layerNodeCount.at(i - 1), 3, dropOutRate, false, true));
		}
		else {
			network.push_back(DropOutLayer(layerNodeCount.at(i), layerNodeCount.at(i - 1), 0, dropOutRate, false, false));
		}
	}
	for (int i = 0; i < network.size(); i++) {
		if (i == 0) {
			network.at(i).setNextLayer(&network.at(i + 1));
		}
		else if (i == network.size() - 1) {
			network.at(i).setPreviousLayer(&network.at(i - 1));
		}
		else {
			network.at(i).setNextLayer(&network.at(i + 1));
			network.at(i).setPreviousLayer(&network.at(i - 1));
		}
	}
	return network;
}
std::vector<double> forwardProp(std::vector<DropOutLayer>& network, std::vector<double>& data) {
	for (int layer = 0; layer < network.size(); layer++) {
		if (layer == 0)
			network.at(layer).setOutput(data);
		else
			network.at(layer).forwardPropagation();
	}
	return network.at(network.size() - 1).getOutput();
}
std::vector<double> forwardProp(std::vector<DropOutLayer>& network, double data) {
	for (int layer = 0; layer < network.size(); layer++) {
		if (layer == 0)
			network.at(layer).setOutput(data);
		else
			network.at(layer).forwardPropagation();
	}
	return network.at(network.size() - 1).getOutput();
}
void backProp(std::vector<DropOutLayer>& network, double loss, double learningRate) {
	for (int layer = 1; layer < network.size(); layer++) {
		network.at(layer).updateAllBiases(loss, learningRate);
		network.at(layer).updateAllWeights(loss, learningRate);
	}
}
void rollNodes(std::vector<DropOutLayer>& network) {
	for (int i = 1; i < network.size(); i++) {
		network.at(i).rollActiveLayers();
	}
}
void testDropOutNetwork(std::vector<DropOutLayer>& dropOutNetwork, int epochs, double learningRate, std::vector<std::vector<double>>& input, std::vector<double>& yTrue, std::string trainingFile, std::string predictFile) {
	std::ofstream dropOutLayerTrainingFile(trainingFile);
	std::ofstream dropOutLayerTestingFile(predictFile);

	double yHat = 0.0;
	double loss = 0;
	double traingingLoss = 0.0;
	double validLoss = 0.0;
	
	int startOfTestIndex = (int)(input.size() * .75);
	int startOfValidIndex = (int)(startOfTestIndex * .75);

	for (int i = 0; i < epochs; i++) {
		traingingLoss = 0;
		for (int j = 0; j < startOfValidIndex; j++) {
			yHat = forwardProp(dropOutNetwork, input[j])[0];
			loss = 2 * (yHat - yTrue[j]);
			traingingLoss += std::pow((yHat - yTrue[j]), 2);
			backProp(dropOutNetwork, loss, learningRate);
			rollNodes(dropOutNetwork);
		}
		validLoss = 0;
		for (int j = startOfValidIndex; j < startOfTestIndex; j++) {
			yHat = forwardProp(dropOutNetwork, input[j])[0];
			validLoss += std::pow((yHat - yTrue[j]), 2);
			rollNodes(dropOutNetwork);
		}
		std::cout << std::fixed << std::setprecision(7) << "Epoch: " << i << "\t" << "Training Loss: " << traingingLoss / startOfValidIndex << "\t" << "Valid Loss: " << validLoss / (startOfTestIndex - startOfValidIndex) << "\t" << std::endl;
		dropOutLayerTrainingFile << std::fixed << std::setprecision(7) << traingingLoss / startOfValidIndex << "," << validLoss / (startOfTestIndex - startOfValidIndex) << "\n";
	}

	loss = 0;
	double yHatOnce = 0;
	double onceLoss = 0;
	for (int j = startOfTestIndex; j < input.size(); j++) {
		rollNodes(dropOutNetwork);
		yHatOnce = forwardProp(dropOutNetwork, input[j])[0];
		yHat = yHatOnce;
		for (int k = 0; k < 2; k++) {
			rollNodes(dropOutNetwork);
			yHat += forwardProp(dropOutNetwork, input[j])[0];
		}
		yHat /= 3;
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
void resetNetworkWeightsAndBiases(std::vector<DropOutLayer>& dropOutNetwork) {
	for (int i = 1; i < dropOutNetwork.size(); i++)
		dropOutNetwork.at(i).resetWeightsAndBias();
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
