/**
 * @file	NetworkIO.h
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

#ifndef NETWORKIO_H_
#define NETWORKIO_H_

#include <string>

#include "NvInfer.h"
#include "NvUtils.h"

namespace jetson_tensorrt {

/**
 * @brief Abstract class representing a node in a neural network which takes input or generates output
 */
class NetworkIO {
public:
	/**
	 * @brief	Creates a new NetworkIO object
	 * @param	name	Name of the network layer
	 * @param	dims	Dimensions of the network layer
	 * @param	eleSize	Size of each individual dimension element
	 */
	NetworkIO(std::string name, nvinfer1::Dims dims, size_t eleSize);

	/**
	 * @brief	NetworkIO destructor
	 */
	virtual ~NetworkIO();

	/**
	 * @brief	Returns the size in bytes of the network layer
	 * @returns	Size sum in bytes of every element
	 */
	size_t size();

	std::string name;
	nvinfer1::Dims dims;
	size_t eleSize;
};

/**
 * @brief Represents an input node in a neural network which accepts a certain dimension and size
 */
class NetworkInput: public NetworkIO {
public:
	/**
	 * @brief	Creates a new NetworkInput object
	 * @param	name	Name of the network input layer
	 * @param	dims	Dimensions of the network input layer
	 * @param	eleSize	Size of each individual dimension element
	 */
	NetworkInput(std::string name, nvinfer1::Dims dims, size_t eleSize);

	/**
	 * @brief	NetworkInput destructor
	 */
	virtual ~NetworkInput();
};

/**
 * @brief Represents an output node in a neural network which outputs a certain dimension and size
 */
class NetworkOutput: public NetworkIO {
public:
	/**
	 * @brief	Creates a new NetworkOutput object
	 * @param	name	Name of the network output layer
	 * @param	dims	Dimensions of the network output layer
	 * @param	eleSize	Size of each individual dimension element
	 */
	NetworkOutput(std::string name, nvinfer1::Dims dims, size_t eleSize);

	/**
	 * @brief	NetworkOutput destructor
	 */
	virtual ~NetworkOutput();
};

}

#endif /* NETWORKIO_H_ */