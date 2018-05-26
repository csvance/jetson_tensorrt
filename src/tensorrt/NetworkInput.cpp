/*
 * NetworkInput.cpp
 *
 *  Created on: May 26, 2018
 *      Author: cvance
 */

#include <string>

#include "NetworkInput.h"

NetworkInput::NetworkInput(std::string name, nvinfer1::Dims dims, size_t eleSize) : NetworkIO(name, dims, eleSize) {}
NetworkInput::~NetworkInput() {}

