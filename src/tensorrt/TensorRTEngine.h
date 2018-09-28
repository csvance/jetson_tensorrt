/**
 * @file	TensorRTEngine.h
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

#ifndef RTENGINE_H_
#define RTENGINE_H_

#include <cuda_runtime_api.h>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "NvUtils.h"

#include "CUDACommon.h"

namespace jetson_tensorrt {

/**
 * @brief Abstract class representing a node in a neural network which takes
 * input or generates output
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
 * @brief Represents an input node in a neural network which accepts a certain
 * dimension and size
 */
class NetworkInput : public NetworkIO {
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
 * @brief Represents an output node in a neural network which outputs a certain
 * dimension and size
 */
class NetworkOutput : public NetworkIO {
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

/**
 * @brief Logger for GIE info/warning/errors
 */
class Logger : public nvinfer1::ILogger {
public:
  void log(nvinfer1::ILogger::Severity, const char *) override;
};

/**
 * @brief Represents all of a batches inputs or outputs located either in host,
 * device, or mapped memory
 */
struct LocatedExecutionMemory {

  LocatedExecutionMemory() {}

  /**
   * @brief LocatedExecutionMemory constructor
   * @param location	The location of the batch inputs or outputs, either in
   * HOST, MAPPED, or DEVICE memory
   * @param batch	Batch of inputs or outputs
   */
  LocatedExecutionMemory(MemoryLocation location,
                         std::vector<std::vector<void *>> batch) {
    this->batch = batch;
    this->location = location;
  }

  MemoryLocation location;
  std::vector<std::vector<void *>> batch;

  std::vector<void *> &operator[](const int index) { return batch[index]; }

  /**
   * @brief Gets the number of units in a batch
   * @return	Number of units in a batch
   */
  size_t size() { return batch.size(); }

  /**
   * @brief Frees the allocated memory
   */
  void release() {

    for (int b = 0; b < batch.size(); b++) {
      for (int c = 0; c < batch[b].size(); c++) {

        if (location == MemoryLocation::HOST) {

          free(batch[b][c]);

        } else if (location == MemoryLocation::DEVICE) {

          cudaError_t deviceFreeError = cudaFree(batch[b][c]);
          if (deviceFreeError != 0) {
            throw std::runtime_error(
                "Error freeing device memory. CUDA Error: " +
                std::to_string(deviceFreeError));
          }

        } else if (location == location == MemoryLocation::MAPPED) {
          cudaError_t hostFreeError = cudaFreeHost(batch[b][c]);
          if (hostFreeError != 0) {
            throw std::runtime_error("Error freeing host memory. CUDA Error: " +
                                     std::to_string(hostFreeError));
          }
        }
      }
    }
  }
};

/**
 * @brief Abstract base class which loads and manages a TensorRT model which
 * hides device/host memory management
 */
class TensorRTEngine {
public:
  /**
   * @brief	Creates and manages a new instance of TensorRTEngine
   */
  TensorRTEngine();

  /**
   * @brief	TensorRTEngine destructor
   */
  virtual ~TensorRTEngine();

  /**
   * @brief	Does a forward pass of the neural network loaded in TensorRT
   * @usage	Should be called after loading the graph and calling
   * allocGPUBuffer()
   * @param	inputs Graph inputs indexed by [batchIndex][inputIndex]
   * @param	outputs Graph inputs indexed by [batchIndex][inputIndex]
   */
  void predict(LocatedExecutionMemory &inputs, LocatedExecutionMemory &outputs);

  /**
   * @brief	Quick load the TensorRT optimized network
   * @usage	Should be called after addInput and addOutput without calling
   * loadModel
   * @param	cachePath	Path to the network cache file
   * @param	maxBatchSize	The max batch size of the saved network. If the
   * batch size needs to be changed, the network should be rebuilt with the new
   * size and not simply changed here.
   */
  void loadCache(std::string cachePath, size_t maxBatchSize = 1);

  /**
   * @brief	Save the TensorRT optimized network for quick loading in the
   * future
   * @usage	Should be called after loadModel()
   * @param	cachePath	Path to the network cache file
   */
  void saveCache(std::string cachePath);

  /**
   * @brief	Returns a summary of the loaded network, inputs, and outputs
   * @return	String containing the summary
   */
  std::string engineSummary();

  /**
   * @brief	Abstract method which registers an input to the network.
   * @param	layer	The name of the input layer (i.e. "input_1")
   * @param	dims	Dimensions of the input layer. Must be in CHW format.
   * Ex: (3, 640, 480)
   * @param	eleSize	Size of each element in bytes
   */
  virtual void addInput(std::string layerName, nvinfer1::Dims dims,
                        size_t eleSize) = 0;

  /**
   * @brief	Abstract method which registers an output to the network.
   * @param	layer	The name of the input layer (i.e. "input_1")
   * @param	dims	Dimension of outputs
   * @param	eleSize	Size of each element in bytes
   */
  virtual void addOutput(std::string layerName, nvinfer1::Dims dims,
                         size_t eleSize) = 0;

  /**
    @brief Allocates a located execution memory structure for inputs
    @param location Whether the memory should be allocated on the HOST, DEVICE,
    or MAPPED
    @param skipMalloc Create the input structure but do not allocate memory
    @return The allocated or mapped inputs
  */
  LocatedExecutionMemory allocInputs(MemoryLocation location,
                                     bool skipMalloc = false);

  /**
    @brief Allocates a located execution memory structure for outputs
    @param location Whether the memory should be allocated on the HOST, DEVICE,
    or MAPPED
    @param skipMalloc Create the input structure but do not allocate memory
    @return The allocated or mapped outputs
   */
  LocatedExecutionMemory allocOutputs(MemoryLocation location,
                                      bool skipMalloc = false);

  int maxBatchSize;
  int numBindings;

  nvinfer1::DataType dataType;

  std::vector<NetworkInput> networkInputs;
  std::vector<NetworkOutput> networkOutputs;

protected:
  nvinfer1::ICudaEngine *engine;
  nvinfer1::IExecutionContext *context;
  Logger logger;

private:
  std::vector<void *> preAllocatedGPUBuffers;

  /**
   * @brief	Allocates the buffers required to copy batches to and from the
   * GPU
   * @usage	Should be called before the first prediction from host memory if
   * not using mapped memory
   */
  void allocGPUBuffer();

  /**
   * @brief	Frees buffers required to copy batches to the GPU
   */
  void freeGPUBuffer();

  bool gpuBufferPreAllocated;
};

} // namespace jetson_tensorrt
#endif
