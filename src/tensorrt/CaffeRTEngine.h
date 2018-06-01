/**
 * @file	CaffeRTEngine.h
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

#ifndef CAFFERTENGINE_H_
#define CAFFERTENGINE_H_

#include <string>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "NvUtils.h"

#include "TensorRTEngine.h"

namespace jetson_tensorrt {

/**
 * @brief Loads and manages a Caffe network in TensorRT
 */
class CaffeRTEngine: public TensorRTEngine {
public:

	/**
	 * @brief	Creates a new instance of CaffeRTEngine
	 */
	CaffeRTEngine();

	/**
	 * @brief	CaffeRTEngine Destructor
	 */
	virtual ~CaffeRTEngine();

	/**
	 * @brief	Loads a trained Caffe model.
	 * @usage	Should be called after registering inputs and outputs with addInput() and addOutput().
	 * @param	prototextPath	Path to the models prototxt file
	 * @param	modelPath	Path to the .caffemodel file
	 * @param	maxBatchSize	The maximum number of records to run a forward network pass on. For maximum performance, this should be the only batch size passed to the network.
	 * @param	dataType	The data type of the network to load into TensorRT
	 * @param	maxNetworkSize	Maximum amount of GPU RAM the Tensorflow graph is allowed to use
	 */
	void loadModel(std::string prototextPath, std::string modelPath, size_t maxBatchSize = 1,
			nvinfer1::DataType dataType = nvinfer1::DataType::kFLOAT,
			size_t maxNetworkSize = (1 << 30));

	/**
	 * @brief	Registers an input to the Caffe network.
	 * @usage	Must be called before loadModel().
	 * @param	layer	The name of the input layer (i.e. "input_1")
	 * @param	dims	Dimensions of the input layer in CHW format
	 * @param	eleSize	Size of each element in bytes
	 */
	void addInput(std::string layer, nvinfer1::Dims dims, size_t eleSize);

	/**
	 * @brief	Registers an output from the Caffe network.
	 * @usage	Must be called before loadModel().
	 * @param	layer	The name of the input layer (i.e. "input_1")
	 * @param	dims	Dimension of outputs
	 * @param	eleSize	Size of each element in bytes
	 */
	void addOutput(std::string layer, nvinfer1::Dims dims, size_t eleSize);

private:
	nvcaffeparser1::ICaffeParser* parser;
};

}

#endif /* CAFFERTENGINE_H_ */
