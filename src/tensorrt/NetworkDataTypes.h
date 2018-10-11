/**
 * @file	NetworkDataTypes.h
 * @author	Carroll Vance
 * @brief	Neural network input and output data types
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

#ifndef TENSORRT_NETWORKIOTYPES_H_
#define TENSORRT_NETWORKIOTYPES_H_

/**
 * @brief	Represents a classified region of an image with a zero indexed
 * class ID and probability value
 */
struct RTClassifiedRegionOfInterest {

  RTClassifiedRegionOfInterest(unsigned int id, float confidence, size_t x,
                               size_t y, size_t w, size_t h) {
    this->id = id;
    this->confidence = confidence;
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
  float confidence;

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
 * @brief Represents the probability that a specific class was in a prediction
 * input
 */
struct RTClassification {

  RTClassification(unsigned int id, float confidence) {
    this->id = id;
    this->confidence = confidence;
  }

  /**
   * @brief	Zero Indexed Class ID
   */
  unsigned int id;

  /**
   * @brief	Confidence that a specific class was in the input
   */
  float confidence;
};

#endif /* TENSORRT_NETWORKIOTYPES_H_ */
