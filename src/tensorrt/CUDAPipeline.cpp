/**
 * @file	CUDAPipeline.cpp
 * @author	Carroll Vance
 * @brief	Abstract class representing a CUDA input / output processing
 * pipeline
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

#include "CUDANode.h"
#include "CUDANodeIO.h"
#include "CUDAPipeline.h"

namespace jetson_tensorrt {

void CUDAPipeline::addNode(CUDANode node) { nodes.push_back(node); }

CUDANodeIO CUDAPipeline::pipe(CUDANodeIO input) {
  CUDANodeIO result = input;

  // Move through the pipe
  for (std::vector<int>::iterator it = nodes.begin(); it != myvector.end();
       ++it)
    result = (*it)->pipe(result);

  return result;
}

static CUDAPipeline CUDAPipeline::createNV12ImageNetPipeline(int inputWidth,
                                                             int inputHeight,
                                                             int outputWidth,
                                                             int outputHeight,
                                                             float3 mean) {
  ToDevicePTRNode ptrNode = ToDevicePTRNode();
  NV12toRGBAfNode nv12Node = NV12toRGBAfNode(inputWidth, inputHeight);
  RGBAfToImageNetNode imgNetNode = RGBAfToImageNetNode(
      inputWidth, inputHeight, outputWidth, outputHeight, mean);

  CUDAPipeline pipe = new CUDAPipeline();
  pipe.addNode(ptrNode);
  pipe.addNode(nv12Node);
  pipe.addNode(imgNetNode);

  return pipe;
}

static CUDAPipeline CUDAPipeline::createRGBImageNetPipeline(int inputWidth,
                                                             int inputHeight,
                                                             int outputWidth,
                                                             int outputHeight,
                                                             float3 mean) {
  ToDevicePTRNode ptrNode = ToDevicePTRNode();
  RGBtoRGBAfNode rbgNode = RGBtoRGBAfNode(inputWidth, inputHeight);
  RGBAfToImageNetNode imgNetNode = RGBAfToImageNetNode(
      inputWidth, inputHeight, outputWidth, outputHeight, mean);

  CUDAPipeline pipe = new CUDAPipeline();
  pipe.addNode(ptrNode);
  pipe.addNode(rgbNode);
  pipe.addNode(imgNetNode);

  return pipe;
}

static CUDAPipeline CUDAPipeline::createRGBAfImageNetPipeline(int inputWidth,
                                                             int inputHeight,
                                                             int outputWidth,
                                                             int outputHeight,
                                                             float3 mean) {
  ToDevicePTRNode ptrNode = ToDevicePTRNode();
  RGBAfToImageNetNode imgNetNode = RGBAfToImageNetNode(
      inputWidth, inputHeight, outputWidth, outputHeight, mean);

  CUDAPipeline pipe = new CUDAPipeline();
  pipe.addNode(ptrNode);
  pipe.addNode(imgNetNode);

  return pipe;
}

} // namespace jetson_tensorrt
