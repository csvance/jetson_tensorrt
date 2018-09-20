/**
 * @file	TensorflowRTEngine.h
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

#ifndef TFRTENGINE_H_
#define TFRTENGINE_H_

#include <string>
#include <vector>

#include "NvUffParser.h"

#include "TensorRTEngine.h"

namespace jetson_tensorrt {

/**
 * @brief Loads and manages a Tensorflow network in TensorRT
 */
class TensorflowRTEngine : public TensorRTEngine {
public:
  /**
   * @brief	Creates a new instance of TensorflowRTEngine
   */
  TensorflowRTEngine();

  /**
   * @brief	TensorflowRTEngine Destructor
   */
  virtual ~TensorflowRTEngine();

  /**
   * @brief	Loads a frozen Tensorflow .uff graph.
   * @usage	Must be called after addInput() and addOutput().
   * @param	uffFile	Path to the .uff graph file
   * @param	maxBatchSize	The maximum number of records to run a forward
   * network pass on. For maximum performance, this should be the only batch
   * size passed to the network.
   * @param	dataType	The data type of the network to load into
   * TensorRT
   * @param	maxNetworkSize	Maximum amount of GPU RAM the Tensorflow graph is
   * allowed to use
   */
  void loadModel(std::string uffFile, size_t maxBatchSize = 1,
                 nvinfer1::DataType dataType = nvinfer1::DataType::kFLOAT,
                 size_t maxNetworkSize = (1 << 30));

  /**
   * @brief	Registers an input to the Tensorflow network. Assumes inputs have
   * no channel dimension.
   * @usage	Must be called before loadUff().
   * @param	layer	The name of the input layer (i.e. "input_1")
   * @param	dims	Dimensions of the input layer.
   * @param	eleSize	Size of each element in bytes
   */
  void addInput(std::string layer, nvinfer1::Dims dims, size_t eleSize);

  /**
   * @brief	Registers an input to the Tensorflow network with a channel
   * dimension and order.
   * @usage	Must be called before loadUff().
   * @param	layer	The name of the input layer (i.e. "input_1")
   * @param	dims	Dimensions of the input layer.
   * @param	eleSize	Size of each element in bytes
   * @param	order Input channel order
   */
  void addInput(std::string layer, nvinfer1::Dims dims, size_t eleSize,
                nvuffparser::UffInputOrder order);

  /**
   * @brief	Registers an output from the Tensorflow network.
   * @usage	Must be called before loadUff().
   * @param	layer	The name of the input layer (i.e. "input_1")
   * @param	dims	Dimension of outputs
   * @param	eleSize	Size of each element in bytes
   */
  void addOutput(std::string layer, nvinfer1::Dims dims, size_t eleSize);

private:
  nvuffparser::IUffParser *parser;
};

} // namespace jetson_tensorrt

#endif /* TFRTENGINE_H_ */
