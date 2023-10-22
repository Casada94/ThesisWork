//
// Created by metal on 10/21/2023.
//

#ifndef NETWORKEXPERIMENTS_DATAPREP_H
#define NETWORKEXPERIMENTS_DATAPREP_H
#include <string>
#include <vector>
void readDataSet(std::string filename, std::vector<std::vector<double>>& input);
void readDataSet(std::string filename, std::vector<double>& yTrue);
std::vector<std::string> splitString(const std::string& str, char delimiter);
void shuffleRows(std::vector<std::vector<double>>& data, std::vector<double>& yTrue, int size);
#endif //NETWORKEXPERIMENTS_DATAPREP_H
