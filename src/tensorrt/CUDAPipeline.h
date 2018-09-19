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

namespace jetson_tensorrt {

class CUDAPipeline {

public:
  CUDAPipeline();

  /**
     @brief Add a node to the pipeline
     @param node The node
   */
  void addNode(CUDANode node);

  /**
     @brief Send an input through the pipeline and get back the output
     @param input The input to the pipeline
     @return The output of the pipeline
   */
  CUDANodeIO pipe(CUDANodeIO &input);

  /**
  @brief Create an NV12 -> ImageNet preprocessing pipeline
  @param inputWidth NV12 image width
  @param inputHeight NV12 image height
  @param outputWidth BGR image width
  @param outputHeight BGR image height
  @param mean ImageNet mean
  @return CUDAPipeline configured to do the required image preprocessing
  */
  static CUDAPipeline CUDAPipeline::createNV12ImageNetPipeline(int inputWidth,
                                                               int inputHeight,
                                                               int outputWidth,
                                                               int outputHeight,
                                                               float3 mean);
  /**
  @brief Create an RGB -> ImageNet preprocessing pipeline
  @param inputWidth RGB image width
  @param inputHeight RGB image height
  @param outputWidth BGR image width
  @param outputHeight BGR image height
  @param mean ImageNet mean
  @return CUDAPipeline configured to do the required image preprocessing
  */
  static CUDAPipeline CUDAPipeline::createRGBImageNetPipeline(int inputWidth,
                                                              int inputHeight,
                                                              int outputWidth,
                                                              int outputHeight,
                                                              float3 mean);

  /**
  @brief Create an RGBAf -> ImageNet preprocessing pipeline
  @param inputWidth RGBAf image width
  @param inputHeight RGBAf image height
  @param outputWidth BGR image width
  @param outputHeight BGR image height
  @param mean ImageNet mean
  @return CUDAPipeline configured to do the required image preprocessing
  */
  static CUDAPipeline
  CUDAPipeline::createRGBAfImageNetPipeline(int inputWidth, int inputHeight,
                                            int outputWidth, int outputHeight,
                                            float3 mean);

  std::vector<CUDANode> nodes;
};

} // namespace jetson_tensorrt

#endif
