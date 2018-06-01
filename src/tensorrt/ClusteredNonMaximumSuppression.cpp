/**
 * @file	ClusteredNonMaximumSuppression.cpp
 * @author	Carroll Vance
 * @brief	Contains a class which handles refining the results of a DetectNet like detector
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

#include <cstddef>
#include <vector>

#include "ClusteredNonMaximumSuppression.h"

namespace jetson_tensorrt {

ClusteredNonMaximumSuppression::ClusteredNonMaximumSuppression() {

	imageDimX = 0;
	imageDimY = 0;
	imageScaleX = 0.0;
	imageScaleY = 0.0;
	inputDimX = 0;
	inputDimY = 0;
	gridDimX = 0;
	gridDimY = 0;
	unitWidth = 0;
	unitHeight = 0;
	gridSize = 0;

	imageReady = false;
	inputReady = false;
	gridReady = false;
}

ClusteredNonMaximumSuppression::~ClusteredNonMaximumSuppression() {}

void ClusteredNonMaximumSuppression::setupImage(size_t imageDimX, size_t imageDimY){
	this->imageDimX = imageDimX;
	this->imageDimY = imageDimY;

	imageReady = true;

	if(imageReady && inputReady && gridReady)
		calculateScale();
}

void ClusteredNonMaximumSuppression::setupInput(size_t inputDimX, size_t inputDimY){
	this->inputDimX = inputDimX;
	this->inputDimY = inputDimY;

	inputReady = true;

	if(imageReady && inputReady && gridReady)
		calculateScale();
}

void ClusteredNonMaximumSuppression::setupGrid(size_t gridDimX, size_t gridDimY){
	this->gridDimX = gridDimX;
	this->gridDimY = gridDimY;
	this->gridSize = gridDimX * gridDimY;

	gridReady = true;

	if(imageReady && inputReady && gridReady)
		calculateScale();
}

void ClusteredNonMaximumSuppression::calculateScale(){
	imageScaleX = imageDimX / gridDimX;
	imageScaleY = imageDimX / gridDimY;

	unitWidth = inputDimX / gridDimX;
	unitHeight = inputDimY / gridDimY;
}

std::vector<ClassRectangle> ClusteredNonMaximumSuppression::execute(float* coverage, float* bboxes, size_t nbClasses, float detectionThreshold){

}


} /* namespace jetson_tensorrt */
