cmake_minimum_required(VERSION 3.26)
project(NetworkExperiments)

set(CMAKE_CXX_STANDARD 17)

add_executable(NetworkExperiments main.cpp
        DropOutLayer.cpp
        FatLayer.cpp
        Layer.cpp)
