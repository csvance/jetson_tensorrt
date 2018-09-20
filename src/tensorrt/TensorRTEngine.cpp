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
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>

#include "NvInfer.h"
#include "NvUtils.h"

using namespace nvinfer1;

#include "CUDACommon.h"
#include "TensorRTEngine.h"

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

  preAllocatedGPUBuffers = std::vector<void *>(maxBatchSize * stepSize);

  /* Allocate GPU Input memory and move input data to it for each batch*/
  for (int b = 0; b < maxBatchSize; b++) {

    /* Allocate GPU Input memory */
    int bindingIdx = 0;
    for (int i = 0; i < networkInputs.size(); i++) {
      size_t inputSize = networkInputs[i].size();

      preAllocatedGPUBuffers[bindingIdx + b * stepSize] =
          safeCudaMalloc(inputSize);
      bindingIdx++;
    }

    /* Allocate GPU Output memory */
    for (int o = 0; o < networkOutputs.size(); o++) {
      size_t outputSize = networkOutputs[o].size();

      preAllocatedGPUBuffers[bindingIdx + b * stepSize] =
          safeCudaMalloc(outputSize);
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
        throw std::runtime_error("Error freeing device memory. CUDA Error: " +
                                 std::to_string(deviceFreeError));
    }

    gpuBufferPreAllocated = false;
  }
}

LocatedExecutionMemory TensorRTEngine::allocInputs(MemoryLocation location,
                                                   bool skipMalloc) {
  LocatedExecutionMemory memory = LocatedExecutionMemory(
      location, std::vector<std::vector<void *>>(maxBatchSize));

  for (int b = 0; b < maxBatchSize; b++) {

    /* Input memory management */
    for (int i = 0; i < networkInputs.size(); i++) {
      size_t inputSize = networkInputs[i].size();
      if (skipMalloc || location == MemoryLocation::NONE) {
        memory[b].push_back(nullptr);
      } else if (location == MemoryLocation::HOST) {
        memory[b].push_back(safeMalloc(inputSize));
      } else if (location == MemoryLocation::DEVICE) {
        memory[b].push_back(safeCudaMalloc(inputSize));
      } else if (location == MemoryLocation::MAPPED) {
        memory[b].push_back(safeCudaHostMalloc(inputSize));
      }
    }
  }

  return memory;
} // namespace jetson_tensorrt

LocatedExecutionMemory TensorRTEngine::allocOutputs(MemoryLocation location,
                                                    bool skipMalloc) {
  LocatedExecutionMemory memory = LocatedExecutionMemory(
      location, std::vector<std::vector<void *>>(maxBatchSize));

  for (int b = 0; b < maxBatchSize; b++) {

    /* Input memory management */
    for (int o = 0; o < networkOutputs.size(); o++) {
      size_t outputSize = networkOutputs[o].size();
      if (skipMalloc || location == MemoryLocation::NONE) {
        memory[b].push_back(nullptr);
      } else if (location == MemoryLocation::HOST) {
        memory[b].push_back(safeMalloc(outputSize));
      } else if (location == MemoryLocation::DEVICE) {
        memory[b].push_back(safeCudaMalloc(outputSize));
      } else if (location == MemoryLocation::MAPPED) {
        memory[b].push_back(safeCudaHostMalloc(outputSize));
      }
    }
  }

  return memory;
} // namespace jetson_tensorrt

void TensorRTEngine::predict(LocatedExecutionMemory &inputs,
                             LocatedExecutionMemory &outputs) {

  if (inputs.size() > maxBatchSize)
    throw std::invalid_argument(
        "Passed batch is larger than maximum batch size");

  int batchCount = inputs.size();
  int stepSize = networkInputs.size() + networkOutputs.size();

  std::vector<void *> transactionGPUBuffers(inputs.size() * stepSize);

  if ((inputs.location == MemoryLocation::HOST ||
       outputs.location == MemoryLocation::HOST) &&
      !gpuBufferPreAllocated)
    allocGPUBuffer();

  /* Assign transaction buffers and copy to GPU if neccisary */
  for (int b = 0; b < batchCount; b++) {
    int bindingIdx = 0;

    /* Input memory management */
    for (int i = 0; i < networkInputs.size(); i++) {
      size_t inputSize = networkInputs[i].size();

      if (inputs.location == MemoryLocation::MAPPED) {

        cudaError_t getDevicePtrError = cudaHostGetDevicePointer(
            &transactionGPUBuffers[bindingIdx + b * stepSize], inputs[b][i], 0);
        if (getDevicePtrError != cudaSuccess) {
          throw std::runtime_error(
              "Unable to get device pointer for CUDA mapped alloc: " +
              std::to_string(getDevicePtrError));
        }

      } else if (inputs.location == MemoryLocation::HOST) {

        // If the batches are in host memory, we need to copy them to the device
        transactionGPUBuffers[bindingIdx + b * stepSize] =
            preAllocatedGPUBuffers[bindingIdx + b * stepSize];

        cudaError_t hostDeviceError =
            cudaMemcpy(transactionGPUBuffers[bindingIdx + b * stepSize],
                       inputs[b][i], inputSize, cudaMemcpyHostToDevice);
        if (hostDeviceError != 0)
          throw std::runtime_error("Unable to copy host memory to device for "
                                   "prediction. CUDA Error: " +
                                   std::to_string(hostDeviceError));

      } else if (inputs.location == MemoryLocation::DEVICE) {

        // Since the inputs are already in the device, all we need to do is
        // assign them to the transaction buffer
        transactionGPUBuffers[bindingIdx + b * stepSize] = inputs[b][i];
      }
      bindingIdx++;
    }

    /* Output memory management */
    for (int o = 0; o < networkOutputs.size(); o++) {

      size_t outputSize = networkOutputs[o].size();

      if (outputs.location == MemoryLocation::MAPPED) {

        cudaError_t getDevicePtrError = cudaHostGetDevicePointer(
            &transactionGPUBuffers[bindingIdx + b * stepSize], outputs[b][o],
            0);
        if (getDevicePtrError != cudaSuccess) {
          throw std::runtime_error(
              "Unable to get device pointer for CUDA mapped alloc: " +
              std::to_string(getDevicePtrError));

        } else if (outputs.location == MemoryLocation::DEVICE)

          // Since the inputs are already in the device, all we need to do is
          // assign them to the transaction buffer
          transactionGPUBuffers[bindingIdx + b * stepSize] = outputs[b][o];

      } else if (outputs.location == MemoryLocation::HOST) {

        transactionGPUBuffers[bindingIdx + b * stepSize] =
            preAllocatedGPUBuffers[bindingIdx + b * stepSize];
      }

      bindingIdx++;
    }
  }

  /* Do the inference */
  if (!context->execute(batchCount, &transactionGPUBuffers[0]))
    throw std::runtime_error(
        "TensorRT engine execution returned unsuccessfully");

  /* Handle outputs for each batch */
  for (int b = 0; b < batchCount; b++) {

    int bindingIdx = inputs[b].size();
    for (int o = 0; o < networkOutputs.size(); o++) {
      size_t outputSize = networkOutputs[o].size();

      if (outputs.location == MemoryLocation::MAPPED) {

        // Don't need to do anything, memory is already mapped

      } else if (outputs.location == MemoryLocation::DEVICE) {

        // Copy the GPU pointer
        outputs[b][o] = transactionGPUBuffers[bindingIdx + b * stepSize];

      } else if (outputs.location == MemoryLocation::HOST) {

        // Transfer memory from GPU back to Host
        cudaError_t deviceHostError = cudaMemcpy(
            outputs[b][o], transactionGPUBuffers[bindingIdx + b * stepSize],
            outputSize, cudaMemcpyDeviceToHost);
        if (deviceHostError != 0)
          throw std::runtime_error("Unable to copy device memory to host for "
                                   "prediction. CUDA Error: " +
                                   std::to_string(deviceHostError));
      }

      bindingIdx++;
    }
  }

} // namespace jetson_tensorrt

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
  // Serialize the loaded model
  nvinfer1::IHostMemory *serMem = engine->serialize();

  if (!serMem)
    throw std::runtime_error("Unable to serialize TensorRT engine");

  // Write the cache file to the disk
  std::ofstream cache;
  cache.open(cachePath.c_str());
  cache.write((const char *)serMem->data(), serMem->size());
  cache.close();

  // Cleanup
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

  nvinfer1::IRuntime *infer = nvinfer1::createInferRuntime(logger);

  gieModelStream.seekg(0, std::ios::end);
  const int modelSize = gieModelStream.tellg();
  gieModelStream.seekg(0, std::ios::beg);

  void *modelMem = malloc(modelSize);
  if (!modelMem)
    throw std::bad_alloc();

  gieModelStream.read((char *)modelMem, modelSize);
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

void Logger::log(nvinfer1::ILogger::Severity severity, const char *msg) {
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


NetworkIO::NetworkIO(std::string name, nvinfer1::Dims dims, size_t eleSize) {
	this->name = name;
	this->dims = dims;
	this->eleSize = eleSize;
}


NetworkIO::~NetworkIO() {
}


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
} // namespace jetson_tensorrt
