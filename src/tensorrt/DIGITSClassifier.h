/**
 * @file	DIGITSClassifier.h
 * @author	Carroll Vance
 * @brief	Loads and manages a DIGITS ImageNet graph with TensorRT
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

#ifndef CLASSIFICATIONRTENGINE_H_
#define CLASSIFICATIONRTENGINE_H_

#include <string>
#include <vector>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "NvUtils.h"

#include "CaffeRTEngine.h"
#include "ImageNetPreprocessor.h"
#include "RTCommon.h"

namespace jetson_tensorrt {

/**
 * @brief	Loads and manages a DIGITS ImageNet graph with TensorRT
 */
class DIGITSClassifier: public CaffeRTEngine {
public:

	/**
	 * @brief	Creates a new instance of DIGITSClassifier
	 * @param	prototextPath	Path to the .prototext file
	 * @param	modelPath	Path to the .caffemodel file
	 * @param	cachePath	Path to the .tensorcache file which will be loaded instead of building the network if present
	 * @param	nbChannels	Number of channels in the input image. 1 for greyscale, 3 for RGB
	 * @param	width	Width of the input image
	 * @param	height	Height of the input image
	 * @param	nbClasses	Number of classes to predict
	 * @param	imageNetMean	The mean value to adjust images to during preprocessing step. 0.0 will disable.
	 * @param	dataType	The data type used to contstruct the TensorRT network. Use FLOAT unless you know how it will effect your model.
	 * @param	maxNetworkSize	Maximum size in bytes of the TensorRT network in device memory
	 */
	DIGITSClassifier(std::string prototextPath, std::string modelPath, std::string cachePath =
			"classification.tensorcache", size_t nbChannels = CHANNELS_BGR, size_t width = 224,
			size_t height = 224, size_t nbClasses = 1, float3 imageNetMean = {
					0.0, 0.0, 0.0 }, nvinfer1::DataType dataType =
					nvinfer1::DataType::kFLOAT, size_t maxNetworkSize = (1 << 30));

	/**
	 * @brief	DIGITSClassifier destructor
	 */
	virtual ~DIGITSClassifier();

	/**
	 * @brief	Classifies a single RBGA format image.
	 * @usage	To prevent memory leakage, the result of classifyRBGA must be deleted after they are no longer needed.
	 * @param	rbga	Pointer to the RBGA image in host memory
	 * @param	width	Width of the image in pixels
	 * @param	height	Height of the input image in pixels
	 * @param	threshold	Minimum probability of a class detection for it to be returned as a result
	 * @param	preprocessOutputAsInput	Don't load memory from the host, instead the output of the last preprocessing operation as the input
	 * @return	Pointer vector of Classification objects above the threshold
	 *
	 */
	std::vector<Classification> classifyRBGAf(float* rbga, size_t width, size_t height, float threshold=0.5, bool preprocessOutputAsInput=false);


	/**
	 * @brief	Classifies in a a single NV12 format image.
	 * @param	nv12	Pointer to the nv12 image in host memory
	 * @param	width	Width of the image in pixels
	 * @param	height	Height of the input image in pixels
	 * @param	threshold	Minimum probability of a class detection for it to be returned as a result
	 * @return	Pointer to a one dimensional array of probabilities for each class
	 *
	 */
	std::vector<Classification> classifyNV12(uint8_t* nv12, size_t width, size_t height, float threshold=0.5);


	static const size_t CHANNELS_GREYSCALE = 1;
	static const size_t CHANNELS_BGR = 3;

	size_t modelWidth;
	size_t modelHeight;
	size_t modelDepth;

	size_t nbClasses;

private:
	static const std::string INPUT_NAME;
	static const std::string OUTPUT_NAME;

	ImageNetPreprocessor* preprocessor;

};

}

#endif /* CLASSIFICATIONRTENGINE_H_ */
