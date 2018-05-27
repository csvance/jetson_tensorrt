/**
 * @file	TensorRTEngine.cpp
 * @author	Carroll Vance
 * @brief	Abstract class which loads and manages a TensorRT graph
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

#include "TensorRTEngine.h"

void* safeCudaMalloc(size_t);
inline int64_t volume(const Dims& d) {
	int64_t v = 1;
	for (int64_t i = 0; i < d.nbDims; i++)
		v *= d.d[i];
	return v;
}

/**
 * @brief	Creates and manages a new instance of TensorRTEngine
 */
TensorRTEngine::TensorRTEngine() {
	numBindings = 0;
	maxBatchSize = 0;
	context = NULL;
	engine = NULL;
}

/**
 * @brief	TensorRTEngine destructor
 */
TensorRTEngine::~TensorRTEngine() {
	/* Clean up */
	context->destroy();
	engine->destroy();
	shutdownProtobufLibrary();
}

/**
 * @brief	Allocates the buffers required to copy batches to the GPU
 * @usage	Should be called by a subclass after determining the number and size of all graph inputs and outputs
 */
void TensorRTEngine::allocGPUBuffer() {
	int stepSize = networkInputs.size() + networkOutputs.size();

	GPU_Buffers = vector<void*>(maxBatchSize * stepSize);

	/* Allocate GPU Input memory and move input data to it for each batch*/
	for (int b = 0; b < maxBatchSize; b++) {

		/* Allocate GPU Input memory */
		int bindingIdx = 0;
		for (int i = 0; i < networkInputs.size(); i++) {
			size_t inputSize = networkInputs[i].size();

			GPU_Buffers[bindingIdx + b * stepSize] = safeCudaMalloc(inputSize);
			bindingIdx++;
		}

		/* Allocate GPU Output memory */
		for (int i = 0; i < networkOutputs.size(); i++) {
			size_t outputSize = networkOutputs[i].size();

			GPU_Buffers[bindingIdx + b * stepSize] = safeCudaMalloc(outputSize);
			bindingIdx++;
		}

	}
}

/**
 * @brief	Frees buffers required to copy batches to the GPU
 * @usage	Should be called by subclass destructor
 */
void TensorRTEngine::freeGPUBuffer() {
	/* Move outputs back to host memory for each batch */
	for (int b = 0; b < GPU_Buffers.size(); b++)
		CHECK(cudaFree(GPU_Buffers[b]));
}

/**
 * @brief	Does a forward pass of the neural network loaded in TensorRT
 * @usage	Should be called after loading the graph and calling allocGPUBuffer()
 * @param	batchInputs Raw graph inputs, indexed by [batchIndex][inputIndex]
 * @return	Raw graph outputs, indexed by [batchIndex][outputNumber]
 */
std::vector<std::vector<void*>> TensorRTEngine::predict(
		std::vector<std::vector<void*>> batchInputs) {

	assert(batchInputs.size() <= maxBatchSize);

	int batchCount = batchInputs.size();
	int stepSize = networkInputs.size() + networkOutputs.size();

	/* Copy batch to GPU */
	for (int b = 0; b < batchCount; b++) {

		int bindingIdx = 0;
		for (int i = 0; i < networkInputs.size(); i++) {
			size_t inputSize = networkInputs[i].size();

			CHECK(
					cudaMemcpy(GPU_Buffers[bindingIdx + b * stepSize],
							batchInputs[b][i], inputSize,
							cudaMemcpyHostToDevice));
			bindingIdx++;
		}

	}

	/* Do the inference */
	assert(context->execute(batchCount, &GPU_Buffers[0]) == true);

	std::vector<std::vector<void*>> Output_Buffers(batchCount);

	/* Move outputs back to host memory for each batch */
	for (int b = 0; b < batchCount; b++) {

		int bindingIdx = batchInputs[b].size();
		for (int i = 0; i < networkOutputs.size(); i++) {
			size_t outputSize = networkOutputs[i].size();

			/* Allocate a host buffer for the network output */
			Output_Buffers[b].push_back(new unsigned char[outputSize]);

			CHECK(
					cudaMemcpy(Output_Buffers[b][i],
							GPU_Buffers[bindingIdx + b * stepSize], outputSize,
							cudaMemcpyDeviceToHost));

			bindingIdx++;
		}

	}

	return Output_Buffers;

}

/**
 * @brief	Returns a summary of the loaded network, inputs, and outputs
 * @return	String containing the summary
 */
string TensorRTEngine::engineSummary() {

	std::stringstream summary;

	for (int i = 0; i < numBindings; ++i) {

		Dims dims = engine->getBindingDimensions(i);
		DataType dtype = engine->getBindingDataType(i);

		summary << "--Binding " << i << "--" << std::endl;
		if (engine->bindingIsInput(i))
			summary << "Type: Input";
		else
			summary << "Type: Output";

		summary << " Dims: (";
		for (int j = 0; j < dims.nbDims; j++)
			summary << dims.d[j] << ",";
		summary << ")" << std::endl;

	}
	return summary.str();

}

/**
 * @brief	Save the TensorRT optimized network for quick loading in the future
 * @usage	Should be called after loadModel()
 * @param	cachePath	Path to the network cache file
 * @return	Whether the save succeeded or not
 */
bool TensorRTEngine::saveCache(std::string cachePath) {
	//Serialize the loaded model
	nvinfer1::IHostMemory* serMem = engine->serialize();

	if (!serMem)
		RETURN_AND_LOG(false, ERROR, "Failed to serialize CUDA engine");

	// Write the cache file to the disk
	std::ofstream cache;
	cache.open(cachePath.c_str());
	cache.write((const char*) serMem->data(), serMem->size());
	cache.close();

	//Cleanup
	serMem->destroy();

	return true;
}

/**
 * @brief	Quick load the TensorRT optimized network
 * @usage	Should be called after addInput and addOutput without calling loadModel
 * @param	cachePath	Path to the network cache file
 * @param	maxBatchSize	The max batch size of the saved network. If the batch size needs to be changed, the network should be rebuilt with the new size and not simply changed here.
 * @return	Whether the load succeeded or not
 */
bool TensorRTEngine::loadCache(std::string cachePath, size_t maxBatchSize) {

	// Read the cache file from the disk and load it into a stringstream
	std::ifstream cache;
	cache.open(cachePath);
	if (!cache.is_open())
		return false;

	std::stringstream gieModelStream;

	gieModelStream.seekg(0, gieModelStream.beg);
	gieModelStream << cache.rdbuf();
	cache.close();

	nvinfer1::IRuntime* infer = nvinfer1::createInferRuntime(logger);

	gieModelStream.seekg(0, std::ios::end);
	const int modelSize = gieModelStream.tellg();
	gieModelStream.seekg(0, std::ios::beg);

	void* modelMem = malloc(modelSize);
	if (!modelMem)
		RETURN_AND_LOG(false, ERROR,
				"Unable to allocate memory to deserialize model");

	gieModelStream.read((char*) modelMem, modelSize);
	engine = infer->deserializeCudaEngine(modelMem, modelSize, NULL);

	free(modelMem);

	if (!engine)
		RETURN_AND_LOG(false, ERROR,
				"Unable to create engine from deserialized data");

	// Populate things that need to be populated before allocating memory
	this->numBindings = engine->getNbBindings();
	this->maxBatchSize = maxBatchSize;

	context = engine->createExecutionContext();

	//Allocate device memory
	allocGPUBuffer();

	return true;
}

void Logger::log(nvinfer1::ILogger::Severity severity, const char* msg) {
	if (severity == Severity::kINFO)
		return;

	switch (severity) {
	case Severity::kINTERNAL_ERROR:
		std::cerr << "INTERNAL_ERROR: ";
		break;
	case Severity::kERROR:
		std::cerr << "ERROR: ";
		break;
	case Severity::kWARNING:
		std::cerr << "WARNING: ";
		break;
	case Severity::kINFO:
		std::cerr << "INFO: ";
		break;
	default:
		std::cerr << "UNKNOWN: ";
		break;
	}
	std::cerr << msg << std::endl;
}

void* safeCudaMalloc(size_t memSize) {
	void* deviceMem;
	CHECK(cudaMalloc(&deviceMem, memSize));
	if (deviceMem == nullptr) {
		std::cerr << "Out of device memory" << std::endl;
		abort();
	}
	return deviceMem;
}
