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

using namespace nvuffparser;
using namespace nvinfer1;

#include "TensorflowRTEngine.h"
#include "host_common.h"


/**
 * @brief	Creates a new instance of TensorflowRTEngine
 */
TensorflowRTEngine::TensorflowRTEngine() : TensorRTEngine() {
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
 * @brief	Registers an input to the Tensorflow network
 * @usage	Must be called before loadUff()
 * @param	layer	The name of the input layer (i.e. "input_1")
 * @param	dims	Dimensions of the input layer in CHW format
 * @param	eleSize	Size of each element in bytes
 */
void TensorflowRTEngine::addInput(string layer, nvinfer1::DimsCHW dims, size_t eleSize) {
	/*
	 Register tensorflow input
	 Even if channel index is last in the data, put it first for TensorRT
	 This network inputs are defined in Keras as follows:
	 Input(shape=(Y, X, C))
	 Where Y = 30, X = 40, C = 1
	 */
	parser->registerInput(layer.c_str(), dims);

	/* Save the size for inferences */
	networkInputs.push_back(volume(dims) * eleSize);
}

/**
 * @brief	Registers an output from the Tensorflow network
 * @usage	Must be called before loadUff()
 * @param	layer	The name of the input layer (i.e. "input_1")
 * @param	dims	Dimension of outputs
 * @param	eleSize	Size of each element in bytes
 */
void TensorflowRTEngine::addOutput(string layer, nvinfer1::Dims dims, size_t eleSize) {
	/*
	 Name of last operation of last non optimizer layer found with
	 `convert_to_uff.py tensorflow --input-file graph.pb -l`
	 A dimension is not neccisary
	 */
	parser->registerOutput(layer.c_str());

	/* Save the size for inferences */
	networkOutputs.push_back(volume(dims) * eleSize);
}

/**
 * @brief	Loads a frozen Tensorflow .uff graph
 * @usage	Must be called after addInput() and addOutput()
 * @param	uffFile	Path to the .uff graph file
 * @param	maximumBatchSize	The maximum number of records to run a forward network pass on. For maximum performance, this should be the only batch size passed to the network.
 * @param	maxNetworkSize	Maximum amount of GPU RAM the Tensorflow graph is allowed to use
 */
bool TensorflowRTEngine::loadUff(const char* uffFile, size_t maximumBatchSize,
		nvinfer1::DataType dataType = nvinfer1::DataType::kFLOAT, size_t maxNetworkSize=(1<<30)) {

	maxBatchSize = maximumBatchSize;

	IBuilder* builder = createInferBuilder(logger);

	INetworkDefinition* network = builder->createNetwork();

	if (dataType == DataType::kFLOAT) {
		if (!parser->parse(uffFile, *network, nvinfer1::DataType::kFLOAT))
			RETURN_AND_LOG(false, ERROR, "Fail to parse");
	} else if (dataType == DataType::kHALF) {
		if (!parser->parse(uffFile, *network, nvinfer1::DataType::kINT8))
			RETURN_AND_LOG(false, ERROR, "Fail to parse");
		builder->setHalf2Mode(true);
	} else if (dataType == DataType::kINT8) {
		if (!parser->parse(uffFile, *network, nvinfer1::DataType::kINT8))
			RETURN_AND_LOG(false, ERROR, "Fail to parse");
		builder->setInt8Mode(true);
	}

	builder->setMaxBatchSize(maxBatchSize);

	builder->setMaxWorkspaceSize(maxNetworkSize);

	engine = builder->buildCudaEngine(*network);
	if (!engine)
		RETURN_AND_LOG(false, ERROR, "Unable to create engine");

	/* Reclaim memory */
	network->destroy();
	builder->destroy();
	parser->destroy();

	context = engine->createExecutionContext();

	numBindings = engine->getNbBindings();

	/* Allocate buffers for inference*/
	allocGPUBuffer();

}

