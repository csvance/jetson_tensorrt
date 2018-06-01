/**
 * @file	ClusteredNonMaximumSuppression.h
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

#ifndef CLUSTEREDNONMAXIMUMSUPPRESSION_H_
#define CLUSTEREDNONMAXIMUMSUPPRESSION_H_

#include <cstddef>
#include <vector>

namespace jetson_tensorrt {

/**
 * @brief	Represents a classified region of an image with a zero indexed class ID and probability value
 */
struct ClassRectangle {

	/**
	 * @brief	Zero Indexed Class ID
	 */
	int id;

	/**
	 * @brief	The confidence of the model's prediction
	 */
	float confidence;

	/**
	 * @brief	X Coordinate in pixels
	 */
	int x;

	/**
	 * @brief	Y Coordinate in pixels
	 */
	int y;

	/**
	 * @brief	Width in pixels
	 */
	int w;

	/**
	 * @brief	Height in pixels
	 */
	int h;
};

/**
 * @brief Class which handles refining the results of a DetectNet like detector
 */
class ClusteredNonMaximumSuppression {
public:

	/**
	 * @brief	Creates a new ClusteredNonMaximumSupression object
	 */
	ClusteredNonMaximumSuppression();

	/**
	 * @brief	ClusteredNonMaximumSuppression destructor
	 */
	virtual ~ClusteredNonMaximumSuppression();

	/**
	 * @brief	Configures the input image dimensions
	 * @usage	Should be called before execute().
	 * @param	imageDimX	X dimension of the entire grid
	 * @param	imageDimY	Y dimension of the entire grid
	 */
	void setupImage(size_t imageDimX, size_t imageDimY);

	/**
	 * @brief	Configures the  network input dimensions
	 * @usage	Should be called before execute().
	 * @param	inputDimX	Width of the original input image
	 * @param	inputDimY	Height of the original input image
	 */
	void setupInput(size_t inputDimX, size_t inputDimY);


	/**
	 * @brief	Configures the output dimensions
	 * @usage	Should be called before execute().
	 * @param	gridDimX	X dimension of the entire grid
	 * @param	gridDimY	Y dimension of the entire grid
	 */
	void setupGrid(size_t gridDimX, size_t gridDimY);

	/**
	 * @brief 	Executes the non maximum suppression.
	 * @usage	Should only be called after setupInput() and setupOutput()
	 * @param	coverage	Coverage Map - Pointer to an n dimensional array of floating point values representing the most dominance of an object in a grid rectangle indexed by [class][y][x]
	 * @param	bboxes	Detection Regions - Pointer to a 4 dimensional array of floating point values representing a rectangle indexed by [x1=0, y1=1, x2=2, y2=3][y][x]
	 * @param	nbClasses	Number of classes in the coverage map
	 * @param	detectionThreshold	The threshold a detection region must have to be considered
	 * @return	A vector of class tagged detection regions
	 */
	std::vector<ClassRectangle> execute(float* coverage, float* bboxes, size_t nbClasses=1, float detectionThreshold=0.5);

private:
	bool imageReady, inputReady, gridReady;
	size_t imageDimX, imageDimY;
	float imageScaleX, imageScaleY;
	size_t inputDimX, inputDimY;
	size_t gridDimX, gridDimY, unitWidth, unitHeight, gridSize;

	void calculateScale();
};

} /* namespace jetson_tensorrt */

#endif /* CLUSTEREDNONMAXIMUMSUPPRESSION_H_ */
