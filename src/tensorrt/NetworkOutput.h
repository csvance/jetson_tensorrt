/*
 * NetworkOutput.h
 *
 *  Created on: May 26, 2018
 *      Author: cvance
 */

#ifndef NETWORKOUTPUT_H_
#define NETWORKOUTPUT_H_

#include "NetworkIO.h"

/**
 * @brief Represents an output node in a neural network which outputs a certain dimension and size
 */
class NetworkOutput: public NetworkIO {
public:
	NetworkOutput(std::string, nvinfer1::Dims, size_t);
	virtual ~NetworkOutput();
};

#endif /* NETWORKOUTPUT_H_ */
