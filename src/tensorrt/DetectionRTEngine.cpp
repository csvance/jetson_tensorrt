/**
 * @file	DetectionRTEngine.cpp
 * @author	Carroll Vance
 * @brief	Loads and manages a DIGITS DetectNet graph with TensorRT
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

#include "DetectionRTEngine.h"
#include "RTExceptions.h"

namespace jetson_tensorrt {

const std::string DetectionRTEngine::INPUT_NAME = "data";
const std::string DetectionRTEngine::OUTPUT_COVERAGE_NAME = "coverage";
const std::string DetectionRTEngine::OUTPUT_BBOXES_NAME = "bboxes";

/**
 * @brief	Creates a new instance of DetectionRTEngine
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
DetectionRTEngine::DetectionRTEngine(std::string prototextPath, std::string modelPath, std::string cachePath,
		size_t nbChannels, size_t width, size_t height, size_t nbClasses,
		size_t maximumBatchSize, nvinfer1::DataType dataType,
		size_t maxNetworkSize) : CaffeRTEngine(){

	if(nbChannels != CHANNELS_BGR)
		throw UnsupportedConfigurationException("Only BGR DetectNets are supported currently");

	addInput(INPUT_NAME, nvinfer1::DimsCHW(nbChannels, height, width), sizeof(float));

	nvinfer1::Dims outputDimsCoverage; outputDimsCoverage.nbDims = 3;
	outputDimsCoverage.d[0] = nbClasses;
	outputDimsCoverage.d[1] = BBOX_DIM_Y;
	outputDimsCoverage.d[2] = BBOX_DIM_X;
	addOutput(OUTPUT_COVERAGE_NAME, outputDimsCoverage, sizeof(float));

	nvinfer1::Dims outputDimsBboxes; outputDimsBboxes.nbDims = 3;
	outputDimsBboxes.d[0] = 4;
	outputDimsBboxes.d[1] = BBOX_DIM_Y;
	outputDimsBboxes.d[2] = BBOX_DIM_X;
	addOutput(OUTPUT_BBOXES_NAME, outputDimsBboxes, sizeof(float));

	try{
		loadCache(cachePath);
	} catch (ModelDeserializeException& e){
		loadModel(prototextPath, modelPath, maximumBatchSize, dataType, maxNetworkSize);
		saveCache(cachePath);
	}
}

DetectionRTEngine::~DetectionRTEngine() {}

} /* namespace jetson_tensorrt */
