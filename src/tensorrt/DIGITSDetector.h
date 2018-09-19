/**
 * @file	DIGITSDetector.h
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

#ifndef DETECTIONRTENGINE_H_
#define DETECTIONRTENGINE_H_

#include <string>
#include <vector>

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "NvUtils.h"

#include "CUDANode.h"
#include "CUDANodeIO.h"
#include "CUDAPipeline.h"
#include "CaffeRTEngine.h"
#include "ClusteredNonMaximumSuppression.h"
#include "ImageNetPreprocessor.h"

namespace jetson_tensorrt {

/**
 * @brief	Loads and manages a DIGITS DetectNet graph with TensorRT
 */
class DIGITSDetector : public CaffeRTEngine {
public:
  /**
   * @brief	Creates a new instance of DIGITSDetector
   * @param	prototextPath	Path to the .prototext file
   * @param	modelPath	Path to the .caffemodel file
   * @param	cachePath	Path to the .tensorcache file which will be loaded
   * instead of building the network if present
   * @param	nbChannels	Number of channels in the input image. 1 for
   * greyscale, 3 for RGB
   * @param	width	Width of the input image
   * @param	height	Height of the input image
   * @param	nbClasses	Number of classes to predict
   * @param	dataType	The data type used to contstruct the TensorRT network.
   * Use FLOAT unless you know how it will effect your model.
   * @param	maxNetworkSize	Maximum size in bytes of the TensorRT network in
   * device memory
   */
  DIGITSDetector(std::string prototextPath, std::string modelPath,
                 std::string cachePath = "detection.tensorcache",
                 size_t nbChannels = CHANNELS_BGR, size_t width = 1024,
                 size_t height = 512, size_t nbClasses = 1,
                 nvinfer1::DataType dataType = nvinfer1::DataType::kFLOAT,
                 size_t maxNetworkSize = (1 << 30));

  /**
   * @brief DIGITSDetector destructor
   */
  virtual ~DIGITSDetector();

  /**
   * @brief	Detects a single BGR format image.
   * @param	inputs Graph inputs indexed by [batchIndex][inputIndex]
   * @param	outputs Graph inputs indexed by [batchIndex][inputIndex]
   * @param	threshold	Minimum coverage threshold for detected regions
   * @returned	vector of ClassRectangles representing the detections
   */
  void predict();
  std::vector<ClassRectangle> DIGITSDetector::detect(LocatedExecutionMemory &inputs,
                                     LocatedExecutionMemory &outputs,
                                     float threshold = 0.5);

  size_t modelWidth;
  size_t modelHeight;
  size_t modelDepth;

  size_t nbClasses;

  static const size_t CHANNELS_GREYSCALE = 1;
  static const size_t CHANNELS_BGR = 3;

private:
  static const std::string INPUT_NAME;
  static const std::string OUTPUT_COVERAGE_NAME;
  static const std::string OUTPUT_BBOXES_NAME;

  static const size_t BBOX_DIM_X = 64;
  static const size_t BBOX_DIM_Y = 32;

  ClusteredNonMaximumSuppression suppressor;
};

} /* namespace jetson_tensorrt */

#endif /* DETECTIONRTENGINE_H_ */
