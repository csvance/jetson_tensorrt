/**
 * @file	ClassificationRTEngine.cpp
 * @author	Carroll Vance
 * @brief	Loads and manages a DIGITS classification graph with TensorRT
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

#include <cassert>
#include <string>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "NvUtils.h"

#include <ClassificationRTEngine.h>

#include "RTExceptions.h"

namespace jetson_tensorrt{

const std::string ClassificationRTEngine::INPUT_BLOB = "data";
const std::string ClassificationRTEngine::OUTPUT_BLOB = "prob";

/**
 * @brief	Creates a new instance of ClassificationRTEngine
 * @param	prototextPath	Path to the .prototext file
 * @param	modelPath	Path to the .caffemodel file
 * @param	cachePath	Path to the .tensorcache file which will be loaded instead of building the network if present
 * @param	nbChannels	Number of channels in the input image. 1 for greyscale, 3 for RGB
 * @param	width	Width of the input image
 * @param	height	Height of the input image
 * @param	nbClasses	Number of classes to predict
 * @param	maximumBatchSize	Maximum number of images that will be passed at once for prediction
 * @param	dataType	The data type used to contstruct the TensorRT network. Use FLOAT unless you know how it will effect your model.
 * @param	maxNetworkSize	Maximum size in bytes of the TensorRT network in device memory
 */
ClassificationRTEngine::ClassificationRTEngine(std::string prototextPath, std::string modelPath, std::string cachePath,
		size_t nbChannels, size_t width, size_t height, size_t nbClasses,
		size_t maximumBatchSize, nvinfer1::DataType dataType,
		size_t maxNetworkSize) : CaffeRTEngine(){

	addInput(INPUT_BLOB, nvinfer1::DimsCHW(nbChannels, height, width), sizeof(float));

	nvinfer1::Dims outputDims; outputDims.nbDims = 1; outputDims.d[0] = nbClasses;
	addOutput(OUTPUT_BLOB, outputDims, sizeof(float));

	try{
		loadCache(cachePath);
	} catch (ModelDeserializeException& e){
		loadModel(prototextPath, modelPath, maximumBatchSize, dataType, maxNetworkSize);
		saveCache(cachePath);
	}
}

/**
 * @brief	ClassificationRTEngine destructor
 */
ClassificationRTEngine::~ClassificationRTEngine() {}

}


