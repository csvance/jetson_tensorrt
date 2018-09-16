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

/**
 * @brief	Allocates CUDA device memory or throws an exception
 * @param	memSize	The size of the requested memory allocation in bytes
 * @return	A handle corresponding to the device memory allocation.
 */
void* safeCudaMalloc(size_t memSize);

/**
 * @brief	Represents a classified region of an image with a zero indexed class ID and probability value
 */
struct ClassRectangle {

	ClassRectangle(unsigned int id, float coverage, size_t x, size_t y, size_t w, size_t h){
		this->id = id;
		this->coverage = coverage;
		this->x = x;
		this->y = y;
		this->w = w;
		this->h = h;
	}

	/**
	 * @brief	Zero Indexed Class ID
	 */
	unsigned int id;

	/**
	 * @brief	The confidence of the model's prediction
	 */
	float coverage;

	/**
	 * @brief	X Coordinate in pixels
	 */
	size_t x;

	/**
	 * @brief	Y Coordinate in pixels
	 */
	size_t y;

	/**
	 * @brief	Width in pixels
	 */
	size_t w;

	/**
	 * @brief	Height in pixels
	 */
	size_t h;
};

/**
 * @brief Represents the probability that a specific class was in a prediction input
 */
struct Classification{

	Classification(unsigned int id, float probability){
		this->id = id;
		this->probability = probability;
	}

	/**
	 * @brief	Zero Indexed Class ID
	 */
	unsigned int id;

	/**
	 * @brief	Probability that a specific class was in the input
	 */
	float probability;
};

/**
 * @brief Represents all of a batches inputs or outputs located either in host or device memory
 */
struct LocatedExecutionMemory {
	enum Location {
		HOST, DEVICE
	};

	/**
	 * @brief LocatedExecutionMemory constructor
	 * @param location	The location of the batch inputs or outputs, either in HOST or DEVICE memory
	 * @param batch	Batch of inputs or outputs
	 */
	LocatedExecutionMemory(Location location,
			std::vector<std::vector<void*> > batch) {
		this->batch = batch;
		this->location = location;
	}

	Location location;
	std::vector<std::vector<void*> > batch;

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
