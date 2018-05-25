/*
 * @file	host_common.h
 * @author	Carroll Vance
 * @brief	Common host functions for interfacing with CUDA devices
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

#ifndef __COMMON_H__
#define __COMMON_H__

#include <iostream>
#include "NvInfer.h"
#include "NvUffParser.h"
#include "NvUtils.h"

using namespace nvuffparser;
using namespace nvinfer1;

inline int64_t volume(const Dims& d) {
	int64_t v = 1;
	for (int64_t i = 0; i < d.nbDims; i++)
		v *= d.d[i];
	return v;
}

void* safeCudaMalloc(size_t);

#define CHECK(status)									\
{														\
	if (status != 0)									\
	{													\
		std::cout << "Cuda failure: " << status;		\
		abort();										\
	}													\
}

#define RETURN_AND_LOG(ret, severity, message)                                              \
    do {                                                                                    \
        std::string error_message = "sample_uff_mnist: " + std::string(message);            \
        logger.log(ILogger::Severity::k ## severity, error_message.c_str());               \
        return (ret);                                                                       \
    } while(0)

// Logger for GIE info/warning/errors
class Logger: public nvinfer1::ILogger {
public:
	void log(nvinfer1::ILogger::Severity, const char*) override;
};

#endif
