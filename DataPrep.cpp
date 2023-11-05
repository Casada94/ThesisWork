//
// Created by metal on 10/21/2023.
//
#include "DataPrep.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <random>


std::vector<std::string> splitString(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::istringstream ss(str);
    std::string token;

    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}
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
                    input[count][i] = ((std::stod(tokens[i])*-1) - 119.5697045) / 2.003531724;
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
