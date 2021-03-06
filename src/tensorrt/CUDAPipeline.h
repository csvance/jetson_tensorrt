/**
 * @file	CUDAPipeline.h
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

#ifndef CUDAPIPELINE_H_
#define CUDAPIPELINE_H_

#include <vector>

#include "CUDACommon.h"

namespace jetson_tensorrt {

class CUDAPipeIO {
public:
  CUDAPipeIO(MemoryLocation location);
  CUDAPipeIO(MemoryLocation location, void *data, size_t dataSize);

  size_t size();

  MemoryLocation location;
  size_t dataSize;
  void *data;
};

class CUDAPipeNode {
public:
  CUDAPipeNode() { this->allocated = false; }
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

class CUDAPipeline {

public:
  CUDAPipeline();
  ~CUDAPipeline();

  /**
     @brief Add a node to the pipeline
     @param node The node
   */
  void addNode(CUDAPipeNode *node);

  /**
     @brief Send an input through the pipeline and get back the output
     @param input The input to the pipeline
     @return The output of the pipeline
   */
  CUDAPipeIO pipe(CUDAPipeIO &input);

  /**
  @brief Create an NV12 -> ImageNet preprocessing pipeline
  @param inputWidth NV12 image width
  @param inputHeight NV12 image height
  @param outputWidth BGR image width
  @param outputHeight BGR image height
  @param mean ImageNet mean
  @return CUDAPipeline configured to do the required image preprocessing
  */
  static CUDAPipeline *
  createNV12ImageNetPipeline(int inputWidth, int inputHeight, int outputWidth,
                             int outputHeight, float3 mean);
  /**
  @brief Create an RGB -> ImageNet preprocessing pipeline
  @param inputWidth RGB image width
  @param inputHeight RGB image height
  @param outputWidth BGR image width
  @param outputHeight BGR image height
  @param mean ImageNet mean
  @return CUDAPipeline configured to do the required image preprocessing
  */
  static CUDAPipeline *createRGBImageNetPipeline(int inputWidth,
                                                 int inputHeight,
                                                 int outputWidth,
                                                 int outputHeight, float3 mean);

  /**
  @brief Create an RGBAf -> ImageNet preprocessing pipeline
  @param inputWidth RGBAf image width
  @param inputHeight RGBAf image height
  @param outputWidth BGR image width
  @param outputHeight BGR image height
  @param mean ImageNet mean
  @return CUDAPipeline configured to do the required image preprocessing
  */
  static CUDAPipeline *
  createRGBAfImageNetPipeline(int inputWidth, int inputHeight, int outputWidth,
                              int outputHeight, float3 mean);

  std::vector<CUDAPipeNode *> nodes;
};

} // namespace jetson_tensorrt

#endif
