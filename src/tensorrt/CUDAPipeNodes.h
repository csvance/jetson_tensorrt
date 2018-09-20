/**
 * @file	CUDAPipeNodes.h
 * @author	Carroll Vance
 * @brief	Nodes for a CUDAPipeline
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

#ifndef CUDANODE_H_
#define CUDANODE_H_

#include "CUDACommon.h"
#include "CUDAPipeline.h"

namespace jetson_tensorrt {

class CUDAPipeNode {
public:
  /**
     @brief Send an input through the node and get back the output
     @param input The input to the node
     @return The output of the node
   */
  virtual CUDAPipeIO pipe(CUDAPipeIO &input) {}

  void *data;
  size_t size();

protected:
  bool allocated;
  size_t allocSize;
  MemoryLocation allocLocation;
};

class ToDevicePTRNode : public CUDAPipeNode {
public:
  ToDevicePTRNode() {}
  virtual ~ToDevicePTRNode() {}

  CUDAPipeIO pipe(CUDAPipeIO &input);
};

class RGBToRGBAfNode : public CUDAPipeNode {
public:
  RGBToRGBAfNode(size_t inputWidth, size_t inputHeight) {
    this->inputWidth = inputWidth;
    this->inputHeight = inputHeight;
  }
  virtual ~RGBToRGBAfNode() {}

  CUDAPipeIO pipe(CUDAPipeIO &input);

  size_t inputWidth, inputHeight;
};

class NV12toRGBAfNode : public CUDAPipeNode {
public:
  NV12toRGBAfNode(size_t inputWidth, size_t inputHeight) {
    this->inputWidth = inputWidth;
    this->inputHeight = inputHeight;
  }
  virtual ~NV12toRGBAfNode();

  CUDAPipeIO pipe(CUDAPipeIO &input);

  size_t inputWidth, inputHeight;
};

class RGBAfToImageNetNode : public CUDAPipeNode {
public:
  RGBAfToImageNetNode(size_t inputWidth, size_t inputHeight, size_t outputWidth,
                      size_t outputHeight, float3 mean) {
    this->inputWidth = inputWidth;
    this->inputHeight = inputHeight;
    this->outputWidth = outputWidth;
    this->outputHeight = outputHeight;
    this->mean = mean;
  }
  virtual ~RGBAfToImageNetNode() {}

  CUDAPipeIO pipe(CUDAPipeIO &input);

  size_t inputWidth, inputHeight;
  size_t outputWidth, outputHeight;
  float3 mean;
};

} // namespace jetson_tensorrt

#endif
