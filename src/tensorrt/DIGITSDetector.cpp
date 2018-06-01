/**
 * @file	DIGITSDetector.cpp
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

#include "DIGITSDetector.h"
#include "RTExceptions.h"

namespace jetson_tensorrt {

const std::string DIGITSDetector::INPUT_NAME = "data";
const std::string DIGITSDetector::OUTPUT_COVERAGE_NAME = "coverage";
const std::string DIGITSDetector::OUTPUT_BBOXES_NAME = "bboxes";


DIGITSDetector::DIGITSDetector(std::string prototextPath, std::string modelPath,
		std::string cachePath, size_t nbChannels, size_t width, size_t height,
		size_t nbClasses, float3 imageNetMean, nvinfer1::DataType dataType,
		size_t maxNetworkSize) :
		CaffeRTEngine() {

	if (nbChannels != CHANNELS_BGR)
		throw UnsupportedConfigurationException(
				"Only BGR DetectNets are supported currently");

	addInput(INPUT_NAME, nvinfer1::DimsCHW(nbChannels, height, width),
			sizeof(float));

	nvinfer1::Dims outputDimsCoverage;
	outputDimsCoverage.nbDims = 3;
	outputDimsCoverage.d[0] = nbClasses;
	outputDimsCoverage.d[1] = BBOX_DIM_Y;
	outputDimsCoverage.d[2] = BBOX_DIM_X;
	addOutput(OUTPUT_COVERAGE_NAME, outputDimsCoverage, sizeof(float));

	nvinfer1::Dims outputDimsBboxes;
	outputDimsBboxes.nbDims = 3;
	outputDimsBboxes.d[0] = 4;
	outputDimsBboxes.d[1] = BBOX_DIM_Y;
	outputDimsBboxes.d[2] = BBOX_DIM_X;
	addOutput(OUTPUT_BBOXES_NAME, outputDimsBboxes, sizeof(float));

	try {
		loadCache(cachePath);
	} catch (ModelDeserializeException& e) {
		loadModel(prototextPath, modelPath, 1, dataType,
				maxNetworkSize);
		saveCache(cachePath);
	}

	this->modelWidth = width;
	this->modelHeight = height;
	this->modelDepth = nbChannels;
	this->nbClasses = nbClasses;

	preprocessor = new ImageNetPreprocessor(imageNetMean);

	//Configure non-maximum suppression based on what we currently know
	suppressor.setupInput(width, height);
	suppressor.setupGrid(BBOX_DIM_X, BBOX_DIM_Y);
}

DIGITSDetector::~DIGITSDetector() {}

std::vector<ClassRectangle> DIGITSDetector::detectRGBA(float* rbga, size_t width, size_t height, float threshold, bool preprocessOutputAsInput){

	if(!preprocessOutputAsInput){
		//Load the image to device
		preprocessor->inputFromHost(rbga, width * height * sizeof(float4));
	}else{
		//Use the ouput from last preprocessor conversion call as input
		preprocessor->swapIO();
	}

	//Convert to BGR
	float* preprocessedImageDevice = preprocessor->RBGAtoBGR(width, height,
			modelWidth, modelHeight);

	//Setup inference
	std::vector<std::vector<void*>> batchInputs(1);
	batchInputs[0].push_back((void*) preprocessedImageDevice);
	LocatedExecutionMemory predictionInputs(LocatedExecutionMemory::DEVICE,
			batchInputs);

	//Execute inference
	LocatedExecutionMemory predictionOutputs = predict(predictionInputs, true);

	float* coverage = (float*) predictionOutputs.batch[0][0];
	float* bboxes = (float*) predictionOutputs.batch[0][1];

	//Configure non-maximum suppressor for the current image size
	suppressor.setupImage(width, height);

	return suppressor.execute(coverage, bboxes, nbClasses, threshold);

}

std::vector<ClassRectangle> DIGITSDetector::detectNV12(uint8_t* nv12, size_t width, size_t height, float threshold){
	//Load the image to device
	preprocessor->inputFromHost(nv12, width * height * 3);

	//Convert to RGBA
	preprocessor->NV12toRGBA(width, height);

	return detectRGBA(nullptr, width, height, true, threshold);
}

} /* namespace jetson_tensorrt */
