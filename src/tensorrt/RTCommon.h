/**
 * @file	RTCommon.h
 * @author	Carroll Vance
 * @brief	Contains common functionality used by the TensoRT layer
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

#ifndef COMMON_H_
#define COMMON_H_

#include <cstddef>
#include <vector>

namespace jetson_tensorrt {

void* safeCudaMalloc(size_t);

/**
 * @brief Represents all of a batches inputs or outputs located either in host or device memory
 */
class LocatedExecutionMemory {
public:
	enum Location {
		HOST, DEVICE
	};

	/**
	 * @brief LocatedExecutionMemory constructor
	 * @param location	The location of the batch inputs or outputs, either in HOST or DEVICE memory
	 * @param batch	Batch of inputs or outputs
	 */
	LocatedExecutionMemory(Location location,
			std::vector<std::vector<void*>> batch) {
		this->batch = batch;
		this->location = location;
	}

	Location location;
	std::vector<std::vector<void*>> batch;

	std::vector<void*>& operator[](const int index) {
		return batch[index];
	}

	/**
	 * @brief Gets the number of units in a batch
	 * @return	Number of units in a batch
	 */
	size_t size() {
		return batch.size();
	}
};

}
#endif /* COMMON_H_ */
