/**
 * @file	NetworkIO.cpp
 * @author	Carroll Vance
 * @brief	Contains classes managing neural network layer dimensions and size
 *
 * Copyright (c) 2018 Carroll Vance.
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <string>

#include "NetworkIO.h"

using namespace nvinfer1;

namespace jetson_tensorrt {

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
NetworkIO::~NetworkIO() {
}

/**
 * @brief	Returns the size in bytes of the network layer
 * @returns	Size sum in bytes of every element
 */
size_t NetworkIO::size() {
	int64_t volume = 1;
	for (int64_t i = 0; i < dims.nbDims; i++)
		volume *= dims.d[i];

	return (size_t) volume * eleSize;
}

NetworkInput::NetworkInput(std::string name, nvinfer1::Dims dims,
		size_t eleSize) :
		NetworkIO(name, dims, eleSize) {
}

NetworkInput::~NetworkInput() {
}

NetworkOutput::NetworkOutput(std::string name, nvinfer1::Dims dims,
		size_t eleSize) :
		NetworkIO(name, dims, eleSize) {
}

NetworkOutput::~NetworkOutput() {
}

}
