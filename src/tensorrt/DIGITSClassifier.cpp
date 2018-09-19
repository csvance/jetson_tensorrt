/**
 * @file	DIGITSClassifier.cpp
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

#include <cassert>
#include <fstream>
#include <string>
#include <vector>

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "NvUtils.h"

#include "DIGITSClassifier.h"

namespace jetson_tensorrt {

const std::string DIGITSClassifier::INPUT_NAME = "data";
const std::string DIGITSClassifier::OUTPUT_NAME = "prob";

DIGITSClassifier::DIGITSClassifier(std::string prototextPath,
                                   std::string modelPath, std::string cachePath,
                                   size_t nbChannels, size_t width,
                                   size_t height, size_t nbClasses,
                                   nvinfer1::DataType dataType,
                                   size_t maxNetworkSize)
    : CaffeRTEngine() {

  addInput(INPUT_NAME, nvinfer1::DimsCHW(nbChannels, height, width),
           sizeof(float));

  nvinfer1::Dims outputDims;
  outputDims.nbDims = 1;
  outputDims.d[0] = nbClasses;
  addOutput(OUTPUT_NAME, outputDims, sizeof(float));

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
}

DIGITSClassifier::~DIGITSClassifier() {}

std::vector<Classification>
DIGITSClassifier::classify(LocatedExecutionMemory &inputs,
                           LocatedExecutionMemory &outputs, float threshold) {

  // Execute inference
  predict(inputs, outputs);

  float *classProbabilities = (float *)outputs[0][0];

  std::vector<Classification> classifications;

  for (int c = 0; c < nbClasses; c++) {
    if (classProbabilities[c] > threshold)
      classifications.push_back(Classification(c, classProbabilities[c]));
  }

  return classifications;
}

} // namespace jetson_tensorrt
