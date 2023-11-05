cmake_minimum_required(VERSION 3.26)
project(NetworkExperiments)

set(CMAKE_CXX_STANDARD 17)

add_executable(NetworkExperiments main.cpp
        DataPrep.cpp
        DropOutLayer.cpp
        Layer.cpp
        GBDropoutLayer.cpp
        MultiLevelLayer.cpp
        DataPrep.cpp
        DenseLayer.cpp)
