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

#define DIGITS_CLASSIFICATION_RT_ENGINE_INPUT "data"
#define DIGITS_CLASSIFICATION_RT_ENGINE_OUTPUT	"prob"

ClassificationRTEngine::ClassificationRTEngine(std::string prototextPath, std::string modelPath, std::string cachePath,
		size_t nbChannels, size_t width, size_t height, size_t nbClasses,
		size_t maximumBatchSize, nvinfer1::DataType dataType,
		size_t maxNetworkSize) : CaffeRTEngine(){

	addInput(DIGITS_CLASSIFICATION_RT_ENGINE_INPUT, nvinfer1::DimsCHW(nbChannels, height, width), sizeof(float));

	nvinfer1::Dims outputDims; outputDims.nbDims = 1; outputDims.d[0] = nbClasses;
	addOutput(DIGITS_CLASSIFICATION_RT_ENGINE_OUTPUT, outputDims, sizeof(float));

	try{
		loadCache(cachePath);
	} catch (ModelDeserializeException& e){
		loadModel(prototextPath, modelPath, maximumBatchSize, dataType, maxNetworkSize);
		saveCache(cachePath);
	}
}

ClassificationRTEngine::~ClassificationRTEngine() {}


