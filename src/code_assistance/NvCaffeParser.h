/*
 * Copyright 1993-2016 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#ifndef NV_CAFFE_PARSER_H
#define NV_CAFFE_PARSER_H

#include <istream>
#include <vector>
#include <unordered_map>
#include <memory>
#include <string>
#include "NvInfer.h"

namespace ditcaffe
{
	class NetParameter;
}

namespace nvcaffeparser1
{

// owned by the parser: goes away when the parser does
class IBlobNameToTensor
{
public:
	virtual nvinfer1::ITensor* find(const char* name) const = 0;
protected:
	virtual ~IBlobNameToTensor() {}
};

class IBinaryProtoBlob
{
public:
	virtual const void*			getData() = 0;
	virtual nvinfer1::DimsNCHW		getDimensions() = 0;
	virtual nvinfer1::DataType	getDataType() = 0;
	virtual void				destroy() = 0;
protected:
	virtual ~IBinaryProtoBlob() {}
};

class IPluginFactory
{
public:
	virtual bool isPlugin(const char* layerName) = 0;
	virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights) = 0;
};


class ICaffeParser
{
public:
	virtual const IBlobNameToTensor*	parse(const char* deploy,
											  const char* model,
											  nvinfer1::INetworkDefinition& network,
											  nvinfer1::DataType weightType) = 0;

	virtual void						setProtobufBufferSize(size_t size) = 0;
	virtual void						setPluginFactory(IPluginFactory* factory) = 0;
	virtual IBinaryProtoBlob*			parseBinaryProto(const char* fileName) = 0;
	virtual void						destroy() = 0;
protected:
	virtual ~ICaffeParser() {}
};

ICaffeParser* createCaffeParser();
void shutdownProtobufLibrary();

}
#endif
