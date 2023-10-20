#ifndef NETWORKEXPERIMENTS_MULTILEVELLAYER_H
#define NETWORKEXPERIMENTS_MULTILEVELLAYER_H

#include <vector>
#include<random>
#include "Layer.h"
//#include <unordered_map>

class MultiLevelLayer: public Layer{
private:
    std::vector<std::vector<double>> weights;
    //weights	[ level in stack of prev layer's node ]
    //			[ level in stack of my layers node]
    //			[ weights from all the prev layers nodes to one of my nodes ]
    //			[ the node i am at now ]
    std::vector<double> bias;
    int levelSize;
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_int_distribution<int> distribution;
    std::uniform_int_distribution<int> distribution2;

public:
    MultiLevelLayer(int nodeCount, int previousLayerNodeCount, int activationFunctionSelected, int levelSize, bool isInputLayer, bool isOutputLayer);
    void resetWeightsAndBias();
    void useAllNodes();
    void rollActiveLayers();

    void updateAllWeights(double loss, double learningRate);
    void updateAllBiases(double loss, double learningRate);
    double getPartDerivThrough(int fromNode, double loss);
    double getPartDerivThrough(int fromNode, int fromNodeStack, double loss);
    void forwardPropagation();
    void shakeWeights(double lowShake, double highShake);
    void setOutput(std::vector<double>& input) override;
};


#endif //NETWORKEXPERIMENTS_MULTILEVELLAYER_H
