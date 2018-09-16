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
#include <stdexcept>

#include "NvInfer.h"
#include "NvUtils.h"

using namespace nvinfer1;

#include "TensorRTEngine.h"
#include "RTCommon.h"

namespace jetson_tensorrt {



TensorRTEngine::TensorRTEngine() {
	numBindings = 0;
	maxBatchSize = 0;
	context = NULL;
	engine = NULL;
	gpuBufferPreAllocated = false;
	dataType = nvinfer1::DataType::kFLOAT;
}


TensorRTEngine::~TensorRTEngine() {
	/* Clean up */
	context->destroy();
	engine->destroy();

	freeGPUBuffer();
}


void TensorRTEngine::allocGPUBuffer() {
	int stepSize = networkInputs.size() + networkOutputs.size();

	preAllocatedGPUBuffers = std::vector<void*>(maxBatchSize * stepSize);

	/* Allocate GPU Input memory and move input data to it for each batch*/
	for (int b = 0; b < maxBatchSize; b++) {

		/* Allocate GPU Input memory */
		int bindingIdx = 0;
		for (int i = 0; i < networkInputs.size(); i++) {
			size_t inputSize = networkInputs[i].size();

			preAllocatedGPUBuffers[bindingIdx + b * stepSize] = safeCudaMalloc(
					inputSize);
			bindingIdx++;
		}

		/* Allocate GPU Output memory */
		for (int i = 0; i < networkOutputs.size(); i++) {
			size_t outputSize = networkOutputs[i].size();

			preAllocatedGPUBuffers[bindingIdx + b * stepSize] = safeCudaMalloc(
					outputSize);
			bindingIdx++;
		}

	}

	gpuBufferPreAllocated = true;
}


void TensorRTEngine::freeGPUBuffer() {

	if (gpuBufferPreAllocated) {

		/* Move outputs back to host memory for each batch */
		for (int b = 0; b < preAllocatedGPUBuffers.size(); b++) {
			cudaError_t deviceFreeError = cudaFree(preAllocatedGPUBuffers[b]);
			if (deviceFreeError != 0)
				throw std::runtime_error(
						"Error freeing device memory. CUDA Error: "
								+ std::to_string(deviceFreeError));
		}

		gpuBufferPreAllocated = false;

	}
}

LocatedExecutionMemory TensorRTEngine::predict(LocatedExecutionMemory& inputs,
		bool copyOutputToHost) {

	if (inputs.size() > maxBatchSize)
		throw std::invalid_argument(
				"Passed batch is larger than maximum batch size");

	int batchCount = inputs.size();
	int stepSize = networkInputs.size() + networkOutputs.size();

	std::vector<void*> transactionGPUBuffers(inputs.size() * stepSize);

	//If this is the first time we are doing a prediction allocate memory for inputs and outputs
	if (!gpuBufferPreAllocated)
		allocGPUBuffer();

	/* Assign transaction buffers and copy to GPU if neccisary */
	for (int b = 0; b < batchCount; b++) {
		int bindingIdx = 0;

		//Inputs
		for (int i = 0; i < networkInputs.size(); i++) {
			size_t inputSize = networkInputs[i].size();

			if (inputs.location == LocatedExecutionMemory::Location::HOST) {
				//If the batches are in host memory, we need to copy them to the device
				transactionGPUBuffers[bindingIdx + b * stepSize] =
						preAllocatedGPUBuffers[bindingIdx + b * stepSize];

				cudaError_t hostDeviceError = cudaMemcpy(
						transactionGPUBuffers[bindingIdx + b * stepSize],
						inputs[b][i], inputSize, cudaMemcpyHostToDevice);
				if (hostDeviceError != 0)
					throw std::runtime_error(
							"Unable to copy host memory to device for prediction. CUDA Error: "
									+ std::to_string(hostDeviceError));
			} else {
				// Since the inputs are already in the device, all we need to do is assign them to the transaction buffer
				transactionGPUBuffers[bindingIdx + b * stepSize] = inputs[b][i];
			}
			bindingIdx++;
		}

		//Outputs
		for (int o = 0; o < networkOutputs.size(); o++) {
			size_t outputSize = networkOutputs[o].size();

			transactionGPUBuffers[bindingIdx + b * stepSize] =
					preAllocatedGPUBuffers[bindingIdx + b * stepSize];

			bindingIdx++;
		}
	}

	/* Do the inference */
	if (!context->execute(batchCount, &transactionGPUBuffers[0]))
		throw std::runtime_error(
				"TensorRT engine execution returned unsuccessfully");

	std::vector<std::vector<void*>> batchOutputs(batchCount);

	/* Handle outputs for each batch */
	for (int b = 0; b < batchCount; b++) {

		int bindingIdx = inputs[b].size();
		for (int i = 0; i < networkOutputs.size(); i++) {
			size_t outputSize = networkOutputs[i].size();

			if (copyOutputToHost) {
				/* Allocate a host buffer for the network output */
				batchOutputs[b].push_back(new unsigned char[outputSize]);

				cudaError_t deviceHostError = cudaMemcpy(batchOutputs[b][i],
						transactionGPUBuffers[bindingIdx + b * stepSize],
						outputSize, cudaMemcpyDeviceToHost);
				if (deviceHostError != 0)
					throw std::runtime_error(
							"Unable to copy device memory to host for prediction. CUDA Error: "
									+ std::to_string(deviceHostError));
			} else {
				batchOutputs[b].push_back(
						transactionGPUBuffers[bindingIdx + b * stepSize]);
			}

			bindingIdx++;
		}

	}

	if (copyOutputToHost) {
		return LocatedExecutionMemory(LocatedExecutionMemory::Location::HOST,
				batchOutputs);
	} else {
		return LocatedExecutionMemory(LocatedExecutionMemory::Location::DEVICE,
				batchOutputs);
	}

}


std::string TensorRTEngine::engineSummary() {

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


void TensorRTEngine::saveCache(std::string cachePath) {
	//Serialize the loaded model
	nvinfer1::IHostMemory* serMem = engine->serialize();

	if (!serMem)
		throw std::runtime_error("Unable to serialize TensorRT engine");

	// Write the cache file to the disk
	std::ofstream cache;
	cache.open(cachePath.c_str());
	cache.write((const char*) serMem->data(), serMem->size());
	cache.close();

	//Cleanup
	serMem->destroy();
}


void TensorRTEngine::loadCache(std::string cachePath, size_t maxBatchSize) {

	// Read the cache file from the disk and load it into a stringstream
	std::ifstream cache;
	cache.open(cachePath);
	if (!cache.is_open())
		throw std::runtime_error("Unable to open network cache file");

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
		throw std::bad_alloc();

	gieModelStream.read((char*) modelMem, modelSize);
	engine = infer->deserializeCudaEngine(modelMem, modelSize, NULL);

	free(modelMem);

	if (!engine)
		throw std::invalid_argument(
				"Unable to create engine from deserialized data");

	// Populate things that need to be populated before allocating memory
	this->numBindings = engine->getNbBindings();
	this->maxBatchSize = maxBatchSize;

	context = engine->createExecutionContext();

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

}
