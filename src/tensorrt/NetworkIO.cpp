/*
 * NetworkIO.cpp
 *
 *  Created on: May 26, 2018
 *      Author: cvance
 */

#include <string>

#include "NetworkIO.h"
#include "HostCommon.h"

using namespace nvinfer1;

/**
 * @brief	Creates a new NetworkIO object
 * @param	name	Name of the network layer
 * @param	dims	Dimensions of the network layer
 * @param	eleSize	Size of each individual dimension element
 */
NetworkIO::NetworkIO(std::string name, nvinfer1::Dims dims, size_t eleSize) {
	this->name = name;
	this->dims = dims;
	this->eleSize = eleSize;
}

/**
 * @brief	NetworkIO destructor
 */
NetworkIO::~NetworkIO() {}

/**
 * @brief	Returns the size in bytes of the network layer
 * @returns	Size sum in bytes of every element
 */
size_t NetworkIO::size(){
	return (size_t)(volume(dims) * eleSize);
}

