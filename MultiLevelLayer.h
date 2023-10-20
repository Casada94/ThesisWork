#ifndef NETWORKEXPERIMENTS_MULTILEVELLAYER_H
#define NETWORKEXPERIMENTS_MULTILEVELLAYER_H

#include "Layer.h"
#include <vector>
#include    <random>

class MultiLevelLayer: public Layer{
private:
    int levelSize;

public:
    MultiLevelLayer(int nodeCount, int previousLayerNodeCount, int activationFunctionSelected, int levelSize, bool isInputLayer, bool isOutputLayer);
    void rollActiveLayers() override;
    void setOutput(std::vector<double>& input) override;
};


#endif //NETWORKEXPERIMENTS_MULTILEVELLAYER_H
