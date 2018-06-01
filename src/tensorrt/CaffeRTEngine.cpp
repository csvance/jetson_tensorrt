/**
 * @file	CaffeRTEngine.cpp
 * @author	Carroll Vance
 * @brief	Loads and manages a Caffe graph with TensorRT
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
#include <cassert>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "NvUtils.h"

#include "CaffeRTEngine.h"
#include "RTExceptions.h"

using namespace nvinfer1;

namespace jetson_tensorrt {

/**
 * @brief	Creates a new instance of CaffeRTEngine
 */
CaffeRTEngine::CaffeRTEngine() :
		TensorRTEngine() {
	parser = nvcaffeparser1::createCaffeParser();
}

/**
 * @brief	CaffeRTEngine Destructor
 */
CaffeRTEngine::~CaffeRTEngine() {
	parser->destroy();
}

/**
 * @brief	Registers an input to the Caffe network.
 * @usage	Must be called before loadModel().
 * @param	layer	The name of the input layer (i.e. "input_1")
 * @param	dims	Dimensions of the input layer in CHW format
 * @param	eleSize	Size of each element in bytes
 */
void CaffeRTEngine::addInput(std::string layer, nvinfer1::Dims dims,
		size_t eleSize) {
	networkInputs.push_back(NetworkInput(layer, dims, eleSize));
}

/**
 * @brief	Registers an output from the Caffe network.
 * @usage	Must be called before loadModel().
 * @param	layer	The name of the input layer (i.e. "input_1")
 * @param	dims	Dimension of outputs
 * @param	eleSize	Size of each element in bytes
 */
void CaffeRTEngine::addOutput(std::string layer, nvinfer1::Dims dims,
		size_t eleSize) {
	networkOutputs.push_back(NetworkOutput(layer, dims, eleSize));
}

/**
 * @brief	Loads a trained Caffe model.
 * @usage	Should be called after registering inputs and outputs with addInput() and addOutput().
 * @param	prototextPath	Path to the models prototxt file
 * @param	modelPath	Path to the .caffemodel file
 * @param	maxBatchSize	The maximum number of records to run a forward network pass on. For maximum performance, this should be the only batch size passed to the network.
 * @param	dataType	The data type of the network to load into TensorRT
 * @param	maxNetworkSize	Maximum amount of GPU RAM the Tensorflow graph is allowed to use
 */
void CaffeRTEngine::loadModel(std::string prototextPath, std::string modelPath,
		size_t maximumBatchSize, nvinfer1::DataType dataType,
		size_t maxNetworkSize) {

	if (networkInputs.size() == 0 || networkOutputs.size() == 0)
		throw ModelDimensionMismatchException(
				"Must add inputs and outputs before calling loadModel");

	this->dataType = dataType;
	this->maxBatchSize = maxBatchSize;

	IBuilder* builder = createInferBuilder(logger);

	INetworkDefinition* network = builder->createNetwork();

	const nvcaffeparser1::IBlobNameToTensor *blobNameToTensor = parser->parse(
			prototextPath.c_str(), modelPath.c_str(), *network, dataType);

	if (dataType == DataType::kFLOAT) {

	} else if (dataType == DataType::kHALF) {
		builder->setHalf2Mode(true);

	} else if (dataType == DataType::kINT8) {
		builder->setInt8Mode(true);
	}

	if (!blobNameToTensor)
		throw ModelBuildException("Failed to parse Caffe network");

	// Register outputs with engine
	for (int n = 0; n < networkOutputs.size(); n++) {
		nvinfer1::ITensor* tensor = blobNameToTensor->find(
				networkOutputs[n].name.c_str());
		network->markOutput(*tensor);
	}

	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(maxNetworkSize);

	engine = builder->buildCudaEngine(*network);
	if (!engine)
		throw ModelBuildException("Failed to create TensorRT engine");

	// Verify registered input shapes match parser
	for (int n = 0; n < networkInputs.size(); n++) {
		nvinfer1::Dims dims = engine->getBindingDimensions(n);

		if (dims.nbDims != networkInputs[n].dims.nbDims)
			throw ModelDimensionMismatchException(
					"Model number of input dimensions do not match what was registered with addInput");

		for (int d = 0; d < dims.nbDims; d++)
			if (dims.d[d] != networkInputs[n].dims.d[d])
				throw ModelDimensionMismatchException(
						"Model input dimensions do not match what was registered with addInput");

	}

	/* Reclaim memory */
	network->destroy();
	builder->destroy();
	parser->destroy();

	context = engine->createExecutionContext();

	numBindings = engine->getNbBindings();

}

}
