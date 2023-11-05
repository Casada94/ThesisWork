#ifndef NETWORKEXPERIMENTS_MULTILEVELLAYER_H
#define NETWORKEXPERIMENTS_MULTILEVELLAYER_H

#include "Layer.h"
#include <vector>
#include    <random>

class MultiLevelLayer: public Layer{
private:
    int levelSize;
    std::vector<int> activeNodes;
    double scalar;
    bool willUseAllNodes;

public:
    MultiLevelLayer(int nodeCount, int levelSize,bool willUseAllNodes, bool isInputLayer,bool isOutputLayer);
    void rollActiveNodes();
    void setOutput(const std::vector<double>& rawInput) override;
    double getPartDerivThrough(int fromNode, double loss) override;
    void forwardPropagation() override;
    std::vector<double>& getOutput() override;
};


#endif //NETWORKEXPERIMENTS_MULTILEVELLAYER_H
