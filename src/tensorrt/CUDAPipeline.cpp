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
#include "CUDAPipeline.h"

namespace jetson_tensorrt {

CUDAPipeIO::CUDAPipeIO(MemoryLocation location) { this->location = location; }

CUDAPipeIO::CUDAPipeIO(MemoryLocation location, void *data, size_t dataSize) {
  this->location = location;
  this->data = data;
  this->dataSize = dataSize;
}

size_t CUDAPipeIO::size() { return dataSize; }

void CUDAPipeline::addNode(CUDAPipeNode* node) { nodes.push_back(node); }

CUDAPipeline::CUDAPipeline(){}
CUDAPipeline::~CUDAPipeline(){
  for (int node=0;node<nodes.size();node++)
    delete nodes[node];
}

CUDAPipeIO CUDAPipeline::pipe(CUDAPipeIO& input) {
  CUDAPipeIO result = input;

  // Move through the pipe
  for (int node=0;node<nodes.size();node++)
    result = nodes[node]->pipe(result);

  return result;
}

CUDAPipeline* CUDAPipeline::createNV12ImageNetPipeline(int inputWidth,
                                                             int inputHeight,
                                                             int outputWidth,
                                                             int outputHeight,
                                                             float3 mean) {
  ToDevicePTRNode* ptrNode = new ToDevicePTRNode();
  NV12toRGBAfNode* nv12Node = new NV12toRGBAfNode(inputWidth, inputHeight);
  RGBAfToImageNetNode* imgNetNode = new RGBAfToImageNetNode(
      inputWidth, inputHeight, outputWidth, outputHeight, mean);

  CUDAPipeline* pipe = new CUDAPipeline();
  pipe->addNode(ptrNode);
  pipe->addNode(nv12Node);
  pipe->addNode(imgNetNode);

  return pipe;
}

CUDAPipeline* CUDAPipeline::createRGBImageNetPipeline(int inputWidth,
                                                             int inputHeight,
                                                             int outputWidth,
                                                             int outputHeight,
                                                             float3 mean) {
  ToDevicePTRNode* ptrNode = new ToDevicePTRNode();
  RGBToRGBAfNode* rbgNode = new RGBToRGBAfNode(inputWidth, inputHeight);
  RGBAfToImageNetNode* imgNetNode = new RGBAfToImageNetNode(
      inputWidth, inputHeight, outputWidth, outputHeight, mean);

  CUDAPipeline* pipe = new CUDAPipeline();
  pipe->addNode(ptrNode);
  pipe->addNode(rbgNode);
  pipe->addNode(imgNetNode);

  return pipe;
}

CUDAPipeline* CUDAPipeline::createRGBAfImageNetPipeline(int inputWidth,
                                                             int inputHeight,
                                                             int outputWidth,
                                                             int outputHeight,
                                                             float3 mean) {
  ToDevicePTRNode* ptrNode = new ToDevicePTRNode();
  RGBAfToImageNetNode* imgNetNode = new RGBAfToImageNetNode(
      inputWidth, inputHeight, outputWidth, outputHeight, mean);

  CUDAPipeline* pipe = new CUDAPipeline();
  pipe->addNode(ptrNode);
  pipe->addNode(imgNetNode);

  return pipe;
}

} // namespace jetson_tensorrt
