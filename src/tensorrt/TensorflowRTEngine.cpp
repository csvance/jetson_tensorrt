/**
 * @file	TensorflowRTEngine.cpp
 * @author	Carroll Vance
 * @brief	Loads and manages a Tensorflow graph with TensorRT
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

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <cassert>
#include <vector>
#include <string>
#include <sstream>

#include "NvInfer.h"
#include "NvUffParser.h"
#include "NvUtils.h"

#include "NetworkInput.h"
#include "NetworkOutput.h"
#include "TensorflowRTEngine.h"
#include "RTExceptions.h"

using namespace nvuffparser;
using namespace nvinfer1;

/**
 * @brief	Creates a new instance of TensorflowRTEngine
 */
TensorflowRTEngine::TensorflowRTEngine() :
		TensorRTEngine() {
	parser = createUffParser();
}

/**
 * @brief	TensorflowRTEngine Destructor
 */
TensorflowRTEngine::~TensorflowRTEngine() {
	parser->destroy();
	freeGPUBuffer();
}

/**
 * @brief	Registers an input to the Tensorflow network.
 * @usage	Must be called before loadUff().
 * @param	layer	The name of the input layer (i.e. "input_1")
 * @param	dims	Dimensions of the input layer. Must be in CHW format. Ex: (3, 640, 480)
 * @param	eleSize	Size of each element in bytes
 */
void TensorflowRTEngine::addInput(std::string layer, nvinfer1::Dims dims,
		size_t eleSize) {
	/*
	 Register tensorflow input
	 Even if channel index is last in the data, put it first for TensorRT
	 This network inputs are defined in Keras as follows:
	 Input(shape=(Y, X, C))
	 Where Y = 30, X = 40, C = 1
	 */

	if(dims.nbDims != 3)
		throw ModelDimensionMismatchException("nVidia requires us to use a 3 channel DimsCHW when defining inputs for networks loaded from .uff files");

	if(dims.d[0] > dims.d[1] || dims.d[0] > dims.d[2])
		throw ModelDimensionMismatchException("In CHW format the channel should always be the first dimension");

	nvinfer1::DimsCHW chwDims = nvinfer1::DimsCHW(dims.d[0], dims.d[1], dims.d[2]);

	parser->registerInput(layer.c_str(), chwDims);

	networkInputs.push_back(NetworkInput(layer, dims, eleSize));
}

/**
 * @brief	Registers an output from the Tensorflow network.
 * @usage	Must be called before loadUff().
 * @param	layer	The name of the input layer (i.e. "input_1")
 * @param	dims	Dimension of outputs
 * @param	eleSize	Size of each element in bytes
 */
void TensorflowRTEngine::addOutput(std::string layer, nvinfer1::Dims dims,
		size_t eleSize) {
	/*
	 Name of last operation of last non optimizer layer found with
	 `convert_to_uff.py tensorflow --input-file graph.pb -l`
	 A dimension is not neccisary
	 */
	parser->registerOutput(layer.c_str());

	networkOutputs.push_back(NetworkOutput(layer, dims, eleSize));
}

/**
 * @brief	Loads a frozen Tensorflow .uff graph.
 * @usage	Must be called after addInput() and addOutput().
 * @param	uffFile	Path to the .uff graph file
 * @param	maximumBatchSize	The maximum number of records to run a forward network pass on. For maximum performance, this should be the only batch size passed to the network.
 * @param	dataType	The data type of the network to load into TensorRT
 * @param	maxNetworkSize	Maximum amount of GPU RAM the Tensorflow graph is allowed to use
 */
void TensorflowRTEngine::loadModel(std::string uffFile, size_t maximumBatchSize,
		nvinfer1::DataType dataType, size_t maxNetworkSize) {

	assert(networkInputs.size() > 0 && networkOutputs.size() > 0);

	maxBatchSize = maximumBatchSize;

	IBuilder* builder = createInferBuilder(logger);

	INetworkDefinition* network = builder->createNetwork();

	if (dataType == DataType::kFLOAT) {
		if (!parser->parse(uffFile.c_str(), *network,
				nvinfer1::DataType::kFLOAT))
			throw ModelBuildException("Failed to parse .uff file");
	} else if (dataType == DataType::kHALF) {
		if (!parser->parse(uffFile.c_str(), *network,
				nvinfer1::DataType::kINT8))
			throw ModelBuildException("Failed to parse .uff file");
		builder->setHalf2Mode(true);
	} else if (dataType == DataType::kINT8) {
		if (!parser->parse(uffFile.c_str(), *network,
				nvinfer1::DataType::kINT8))
			throw ModelBuildException("Failed to parse .uff file");
		builder->setInt8Mode(true);
	}

	builder->setMaxBatchSize(maxBatchSize);

	builder->setMaxWorkspaceSize(maxNetworkSize);

	engine = builder->buildCudaEngine(*network);
	if (!engine)
		throw ModelBuildException("Failed to create TensorRT engine");

	/* Reclaim memory */
	network->destroy();
	builder->destroy();
	parser->destroy();

	context = engine->createExecutionContext();

	numBindings = engine->getNbBindings();

	/* Allocate buffers for inference*/
	allocGPUBuffer();

}

