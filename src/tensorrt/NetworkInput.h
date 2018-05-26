/*
 * NetworkInput.h
 *
 *  Created on: May 26, 2018
 *      Author: cvance
 */

#ifndef NETWORKINPUT_H_
#define NETWORKINPUT_H_

#include <string>

#include "NetworkIO.h"

class NetworkInput: public NetworkIO {
public:
	NetworkInput(std::string, nvinfer1::Dims, size_t);
	virtual ~NetworkInput();
};

#endif /* NETWORKINPUT_H_ */
