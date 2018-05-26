/*
 * NetworkIO.h
 *
 *  Created on: May 26, 2018
 *      Author: cvance
 */

#ifndef NETWORKIO_H_
#define NETWORKIO_H_

#include <string>

#include "NvInfer.h"
#include "NvUtils.h"

class NetworkIO {
public:
	NetworkIO(std::string, nvinfer1::Dims, size_t);
	virtual ~NetworkIO();

	std::string name;
	nvinfer1::Dims dims;
	size_t eleSize;

	size_t size();
};

#endif /* NETWORKIO_H_ */
