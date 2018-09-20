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

#include <fstream>

#include "CUDANode.h"
#include "CUDANodeIO.h"
#include "CUDAPipeline.h"
#include "DIGITSDetector.h"

namespace jetson_tensorrt {

const std::string DIGITSDetector::INPUT_NAME = "data";
const std::string DIGITSDetector::OUTPUT_COVERAGE_NAME = "coverage";
const std::string DIGITSDetector::OUTPUT_BBOXES_NAME = "bboxes";

DIGITSDetector::DIGITSDetector(std::string prototextPath, std::string modelPath,
                               std::string cachePath, size_t nbChannels,
                               size_t width, size_t height, size_t nbClasses,
                               nvinfer1::DataType dataType,
                               size_t maxNetworkSize)
    : CaffeRTEngine() {

  if (nbChannels != CHANNELS_BGR)
    throw std::invalid_argument("Only BGR DetectNets are supported currently");

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

  std::ifstream infile(cachePath);
  if (infile.good()) {
    loadCache(cachePath);
  } else {
    loadModel(prototextPath, modelPath, 1, dataType, maxNetworkSize);
    saveCache(cachePath);
  }

  this->modelWidth = width;
  this->modelHeight = height;
  this->modelDepth = nbChannels;
  this->nbClasses = nbClasses;

  // Configure non-maximum suppression based on what we currently know
  suppressor.setupInput(width, height);
  suppressor.setupGrid(BBOX_DIM_X, BBOX_DIM_Y);
}

DIGITSDetector::~DIGITSDetector() {}

std::vector<ClassRectangle>
DIGITSDetector::detect(LocatedExecutionMemory &inputs,
                       LocatedExecutionMemory &outputs, float threshold) {

  // Execute inference
  predict(inputs, outputs);

  float *coverage = (float *)outputs.batch[0][0];
  float *bboxes = (float *)outputs.batch[0][1];

  // Configure non-maximum suppressor for the current image size
	// TODO: convert bounding rects back to original image scale
  suppressor.setupImage(modelWidth, modelHeight);

  return suppressor.execute(coverage, bboxes, nbClasses, threshold);
}

} /* namespace jetson_tensorrt */
