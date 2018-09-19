/**
 * @file	CUDANode.h
 * @author	Carroll Vance
 * @brief	Abstract class representing a CUDA input / output pipeline node
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
#include "RTCommon.h"

#include "cudaImageNetPreprocess.h"
#include "cudaRGB.h"
#include "cudaYUV.h"

namespace jetson_tensorrt {

ToDevicePTRNode::ToDevicePTRNode() {}

CUDANodeIO ToDevicePTRNode::pipe(CUDANodeIO &input) {
  CUDANodeIO output = CUDANodeIO(MemoryLocation::DEVICE, nullptr, input.size());

  if (input.location == MemoryLocation::DEVICE) {

    output.data = input.data;
    output.location = MemoryLocation::DEVICE;

  } else if (input.location == MemoryLocation::HOST) {

    if (!allocated) {
      data = safeCudaMalloc(output.size());
      allocLocation = MemoryLocation::DEVICE;
      allocSize = output.size();
      allocated = true;
    } else {
      if (output.size() != allocSize)
        throw std::runtime_error(
            "CUDAPipeline does not support variable sized inputs");
    }

    output.data = data;

    cudaError_t hostDeviceError = cudaMemcpy(
        output.data, input.data, input.size(), cudaMemcpyHostToDevice);
    if (hostDeviceError != 0)
      throw std::runtime_error(
          "Unable to copy host memory to device. CUDA Error: " +
          std::to_string(hostDeviceError));

  } else if (input.location == MemoryLocation::MAPPED) {

    cudaError_t getDevicePtrError =
        cudaHostGetDevicePointer(&output.data, input.data, 0);
    if (getDevicePtrError != cudaSuccess) {
      throw std::runtime_error(
          "Unable to get device pointer for CUDA mapped alloc: " +
          std::to_string(getDevicePtrError));
    }
  }

  return output;
}

RGBToRGBAfNode::RGBToRGBAfNode(size_t inputWidth, size_t inputHeight) {
  this->inputWidth = inputWidth;
  this->inputHeight = inputHeight;
}

CUDANodeIO RGBToRGBAfNode::pipe(CUDANodeIO &input) {

  if (input.location != MemoryLocation::DEVICE)
    throw std::runtime_error(
        "RGBToRGBAfNode requires inputs in MemoryLocation::DEVICE");

  CUDANodeIO output = CUDANodeIO(MemoryLocation::DEVICE, nullptr,
                                 inputWidth * inputHeight * sizeof(float4));

  if (!allocated) {
    data = safeCudaMalloc(output.size());
    allocLocation = MemoryLocation::DEVICE;
    allocSize = output.size();
    allocated = true;
  } else {
    if (output.size() != allocSize)
      throw std::runtime_error(
          "CUDAPipeline does not support variable sized inputs");
  }

  output.data = data;

  cudaError_t kernelError = cudaRGBToRGBAf(
      (uchar3 *)input.data, (float4 *)output.data, inputWidth, inputHeight);
  if (kernelError != 0)
    throw std::runtime_error(
        "cudaRGBToRGBAf kernel returned an error. CUDA Error: " +
        std::to_string(kernelError));

  return output;
}

NV12toRGBAfNode::NV12toRGBAfNode(size_t inputWidth, size_t inputHeight) {
  this->inputWidth = inputWidth;
  this->inputHeight = inputHeight;
}

CUDANodeIO NV12toRGBAfNode::pipe(CUDANodeIO &input) {

  if (input.location != MemoryLocation::DEVICE)
    throw std::runtime_error(
        "NV12toRGBAfNode requires inputs in MemoryLocation::DEVICE");

  CUDANodeIO output = CUDANodeIO(MemoryLocation::DEVICE, nullptr,
                                 inputWidth * inputHeight * sizeof(float4));

  if (!allocated) {
    data = safeCudaMalloc(output.size());
    allocLocation = MemoryLocation::DEVICE;
    allocSize = output.size();
    allocated = true;
  } else {
    if (output.size() != allocSize)
      throw std::runtime_error(
          "CUDAPipeline does not support variable sized inputs");
  }

  output.data = data;

  cudaError_t kernelError =
      cudaNV12ToRGBAf((unsigned char *)input.data, (float4 *)output.data,
                      inputWidth, inputHeight);
  if (kernelError != 0)
    throw std::runtime_error(
        "NV12toRGBAfNode kernel returned an error. CUDA Error: " +
        std::to_string(kernelError));

  return output;
}

RGBAfToImageNetNode::RGBAfToImageNetNode(size_t inputWidth, size_t inputHeight,
                                         size_t outputWidth,
                                         size_t outputHeight, float3 mean) {

  this->inputWidth = inputWidth;
  this->inputHeight = inputHeight;
  this->outputWidth = outputWidth;
  this->outputHeight = outputHeight;
  this->mean = mean;
}

CUDANodeIO RGBAfToImageNetNode::pipe(CUDANodeIO &input) {
  if (input.location != MemoryLocation::DEVICE)
    throw std::runtime_error(
        "RGBAfToImageNetNode requires inputs in MemoryLocation::DEVICE");

  CUDANodeIO output =
      CUDANodeIO(MemoryLocation::DEVICE, nullptr,
                 3 * outputWidth * outputHeight * sizeof(float));

  if (!allocated) {
    data = safeCudaMalloc(output.size());
    allocLocation = MemoryLocation::DEVICE;
    allocSize = output.size();
    allocated = true;
  } else {
    if (output.size() != allocSize)
      throw std::runtime_error(
          "CUDAPipeline does not support variable sized inputs");
  }

  output.data = data;

  if (mean.x == 0.0 && mean.y == 0.0 && mean.z == 0.0) {
    cudaError_t kernelError =
        cudaPreImageNet((float4 *)input.data, inputWidth, inputHeight,
                        (float *)output.data, outputWidth, outputHeight);
    if (kernelError != 0)
      throw std::runtime_error(
          "cudaPreImageNet kernel returned an error. CUDA Error: " +
          std::to_string(kernelError));
  } else {
    cudaError_t kernelError = cudaPreImageNetMean(
        (float4 *)input.data, inputWidth, inputHeight, (float *)output.data,
        outputWidth, outputHeight, mean);
    if (kernelError != 0)
      throw std::runtime_error(
          "cudaPreImageNetMean kernel returned an error. CUDA Error: " +
          std::to_string(kernelError));
  }

  return output;
}

} // namespace jetson_tensorrt
