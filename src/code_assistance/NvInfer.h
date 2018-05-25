/*
 * Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
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

#ifndef NV_INFER_H
#define NV_INFER_H

#include <cstddef>
#include <cstdint>
#define NV_TENSORRT_MAJOR 3	//!< TensorRT major version
#define NV_TENSORRT_MINOR 0	//!< TensorRT minor version
#define NV_TENSORRT_PATCH 4	//!< TensorRT patch version

#define NV_GIE_MAJOR NV_TENSORRT_MAJOR	//!< Legacy major version name
#define NV_GIE_MINOR NV_TENSORRT_MINOR	//!< Legacy minor version name
#define NV_GIE_PATCH NV_TENSORRT_PATCH	//!< Legacy patch version name

#define NV_TENSORRT_SONAME_MAJOR 4		//!< shared object library major version number
#define NV_TENSORRT_SONAME_MINOR 0		//!< shared object library minor version number
#define NV_TENSORRT_SONAME_PATCH 4		//!< shared object library patch version number

 /**
  * \mainpage
  * 
  * This is the API documentation for the NVIDIA TensorRT library. It provides information on individual functions, classes
  * and methods. Use the index on the left to navigate the documentation.
  * 
  * Please see the accompanying user guide and samples for higher-level information and general advice on using TensorRT.
  */
 
 /** 
  * \file NvInfer.h
  *
  * This is the top-level API file for TensorRT.
  */

// forward declare some CUDA types to avoid an include dependency

struct cublasContext;
struct cudnnContext;

typedef struct CUstream_st *cudaStream_t;	//!< forward declaration of cudaStream_t
typedef struct CUevent_st *cudaEvent_t;		//!< forward declaration of cudaEvent_t

static const int NV_TENSORRT_VERSION = (NV_TENSORRT_MAJOR * 1000) + (NV_TENSORRT_MINOR * 100) + NV_TENSORRT_PATCH;	// major, minor, patch
static const int NV_GIE_VERSION = NV_TENSORRT_VERSION;	// legacy version name

/**
 * \namespace nvinfer1
 * 
 * \brief The TensorRT API version 1 namespace
 * 
 */
namespace nvinfer1
{

template<typename T> inline int EnumMax();  //!< maximum number of elements in an enumeration type

/**
* \enum DataType
* \brief the type of weights and tensors.
*/
enum class DataType : int
{
	kFLOAT = 0,		//!< FP32 format
	kHALF = 1,		//!< FP16 format
	kINT8 = 2		//!< INT8 format
};
template<> inline int EnumMax<DataType>() { return 3; }		//!< maximum number of elements in DataType enum. \see DataType


/**
 * \enum DimensionType
 * \brief the type of data encoded across this dimension
 */

enum class DimensionType : int
{
	kSPATIAL = 0,			//!< elements correspond to different spatial data
	kCHANNEL = 1,			//!< elements correspond to different channels
	kINDEX = 2,				//!< elements correspond to different batch index
	kSEQUENCE = 3			//!< elements correspond to different sequence values
};

template<> inline int EnumMax<DimensionType>() { return 4; }	//!< maximum number of elements in DimensionType enum. \see DimensionType



/**
 * \class Dims
 * \brief structure to define the dimensions of a tensor
 * 
 * \note: currently the following formats are supported for layer inputs and outputs: 
 * * zero or more index dimensions followed by one channel and two spatial dimensions (e.g. CHW)
 * * one time series dimension followed by one index dimension followed by one channel dimension (i.e. TNC)
 */
class Dims
{
public:
	static const int MAX_DIMS = 8;			//!< the maximum number of dimensions supported for a tensor
	int nbDims;								//!< the number of dimensions 
	int d[MAX_DIMS];					//!< the extent of each dimension
	DimensionType type[MAX_DIMS];			//!< the type of each dimension
};

/** 
 * \class DimsHW
 * \brief descriptor for two-dimensional spatial data
 */

class DimsHW : public Dims
{
public:
	/**
	 * \brief construct an empty DimsHW object
	 */
	DimsHW()
	{
		nbDims = 2;
		type[0] = type[1] = DimensionType::kSPATIAL;
		d[0] = d[1] = 0;
	}

	/** \brief construct a DimsHW given height and width
	 *
	 * \param height the height of the data
	 * \param width the width of the data
	 */
	DimsHW(int height, int width)
	{
		nbDims = 2;
		type[0] = type[1] = DimensionType::kSPATIAL;
		d[0] = height;
		d[1] = width;
	}

	/** \brief get the height
	* 
	* \return the height
	*/
	int& h() { return d[0]; }

	/** \brief get the height
	*
	* \return the height
	*/
	int h() const { return d[0]; }

	/** \brief get the width
	*
	* \return the width
	*/
	int& w() { return d[1]; }

	/** \brief get the width
	*
	* \return the width
	*/
	int w() const { return d[1]; }
};

/**
* \class DimsCHW
* \brief descriptor for data with one channel dimension and two spatial dimensions
*/

class DimsCHW : public Dims
{
public:
	/**
	* \brief construct an empty DimsCHW object
	*/
	DimsCHW()
	{
		nbDims = 3;
		type[0] = DimensionType::kCHANNEL;
		type[1] = type[2] = DimensionType::kSPATIAL;
		d[0] = d[1] = d[2] = 0;
	}

	/** \brief construct a DimsCHW given channel count, height and width
	*
	* \param channels the channel count
	* \param height the height of the data
	* \param width the width of the data
	*/

	DimsCHW(int channels, int height, int width)
	{
		nbDims = 3;
		type[0] = DimensionType::kCHANNEL;
		type[1] = type[2] = DimensionType::kSPATIAL;
		d[0] = channels;
		d[1] = height;
		d[2] = width;
	}

	/** \brief get the channel count
	*
	* \return the channel count
	*/
	int& c() { return d[0]; }

	/** \brief get the channel count
	*
	* \return the channel count
	*/
	int c() const { return d[0]; }

	/** \brief get the height
	*
	* \return the height
	*/
	int& h() { return d[1]; }

	/** \brief get the height
	*
	* \return the height
	*/
	int h() const { return d[1]; }

	/** \brief get the width
	*
	* \return the width
	*/
	int& w() { return d[2]; }
	
	/** \brief get the width
	*
	* \return the width
	*/
	int w() const { return d[2]; }

};

/**
* \class DimsNCHW
* \brief descriptor for data with one index dimension, one channel dimension and two spatial dimensions
*/

class DimsNCHW : public Dims
{
public:
	/**
	* \brief construct an empty DimsNCHW object
	*/
	DimsNCHW()
	{
		nbDims = 4;
		type[0] = DimensionType::kINDEX;
		type[1] = DimensionType::kCHANNEL;
		type[2] = type[3] = DimensionType::kSPATIAL;
		d[0] = d[1] = d[2] = d[3] = 0;
	}

	/** \brief construct a DimsCHW given channel count, height and width
	*
	* \param batchSize the batch size (commonly denoted N)
	* \param channels the channel count
	* \param height the height of the data
	* \param width the width of the data
	*/

	DimsNCHW(int batchSize, int channels, int height, int width)
	{
		nbDims = 4;
		type[0] = DimensionType::kINDEX;
		type[1] = DimensionType::kCHANNEL;
		type[2] = type[3] = DimensionType::kSPATIAL;
		d[0] = batchSize;
		d[1] = channels;
		d[2] = height;
		d[3] = width;
	}

	/** \brief get the index count
	*
	* \return the index count
	*/
	int& n() { return d[0]; }

	/** \brief get the index count
	*
	* \return the index count
	*/
	int n() const { return d[0]; }

	/** \brief get the channel count
	*
	* \return the channel count
	*/
	int& c() { return d[1]; }

	/** \brief get the channel count
	*
	* \return the channel count
	*/
	int c() const { return d[1]; }

	/** \brief get the height
	*
	* \return the height
	*/
	int& h() { return d[2]; }

	/** \brief get the height
	*
	* \return the height
	*/
	int h() const { return d[2]; }

	/** \brief get the width
	*
	* \return the width
	*/
	int& w() { return d[3]; }

	/** \brief get the width
	*
	* \return the width
	*/
	int w() const { return d[3]; }
};



/**
 * \class Weights
 * 
 * \brief an array of weights used as a layer parameter
 * 
 * The weights are held by reference until the engine has been built. Therefore the data referenced
 * by \p values field should be preserved until the build is complete
 */

class Weights
{
public:
	DataType type;				//!< the type of the weights
	const void* values;			//!< the weight values, in a contiguous array
	int64_t count;				//!< the number of weights in the array
};

/**
 * \class IHostMemory
 *
 * \brief class to handle library allocated memory that is accessible to the user.
 *
 * The memory allocated via the host memory object is owned by the library and will
 * be de-allocated when the destroy method is called.
 */
class IHostMemory
{
    public:
        virtual void *data() const = 0; //!< A pointer to the raw data that is owned by the library.
        virtual std::size_t size() const = 0; //!< The size in bytes of the data that was allocated.
        virtual DataType type() const = 0; //!< The type of the memory that was allocated.
        virtual void destroy() = 0; //!< Destroy the allocated memory
    protected:
        virtual ~IHostMemory() {}
};

/**
 * \enum LayerType
 *
 * \brief the type values of layer classes
 *
 * \see ILayer::getType()
 * 
 */


enum class LayerType : int
{
	kCONVOLUTION = 0,				//!< Convolution layer
	kFULLY_CONNECTED = 1,			//!< Fully connected layer
	kACTIVATION = 2,				//!< Activation layer
	kPOOLING = 3,					//!< Pooling layer
	kLRN = 4,						//!< LRN layer
	kSCALE = 5,						//!< Scale Layer
	kSOFTMAX = 6,					//!< SoftMax layer
	kDECONVOLUTION = 7,				//!< Deconvolution layer
	kCONCATENATION = 8,				//!< Concatenation layer
	kELEMENTWISE = 9,				//!< Elementwise layer
    kPLUGIN = 10,				    //!< Plugin layer
    kRNN = 11,                      //!< RNN Layer
	kUNARY = 12, 					//!< UnaryOp Operation Layer
	kPADDING = 13,					//!< Padding Layer
	kSHUFFLE = 14					//!< Shuffle Layer
};

template<> inline int EnumMax<LayerType>() { return 15; }	//!< maximum number of elements in LayerType enum. \see LayerType

/**
* \class ITensor
*
* \brief a tensor in a network definition
*
*/

class ITensor
{
public:

	/** \brief Set the tensor name
	*
	* For a network input, the name is assigned by the application. For tensors which are layer outputs,
	* a default name is assigned consisting of the layer name followed by the index of the output in brackets.
	*
	* This method copies the name string
	*
	* \param name the name
	*
	* \see getName()
	*/

	virtual void setName(const char* name) = 0;

	/** \brief get the tensor name
	 * 
	 * \return the name, as a pointer to a NULL-terminated character sequence
	 *
	 * \see setName()
	 */

	virtual const char* getName() const = 0;


	/** \brief Set the dimensions of a tensor
	*
	* For a network input the name is assigned by the application. For a network output it is computed based on
	* the layer parameters and the inputs to the layer. If a tensor size or a parameter is modified in the network,
	* the dimensions of all dependent tensors will be recomputed.
	*
	* This call is only legal for network input tensors, since the dimensions of layer output tensors are inferred based on
	* layer inputs and parameters.
	*
	* \param dimensions the dimensions of the tensor
	*
	* \see getDimensions()
	*/

	virtual void setDimensions(Dims dimensions) = 0;				// only valid for input tensors


	/** \brief Get the dimensions of a tensor
 	 *
	 * \return the dimensions of the layer
	 *
	 * \see setDimensions()
	 */

	virtual Dims getDimensions() const = 0;

	/** \brief Set the data type of a tensor
	 *
	 * \param type the data type of the tensor
	 *
     * The type is unchanged if the type is
     * invalid for the given tensor.
     *
     * If the tensor is a network input or output,
     * then the tensor type cannot be DataType::kINT8.
     *
	 * \see getType()
	 */
	virtual void setType(DataType type) = 0;

	/** \brief Get the data type of a tensor
	 * 
	 * \return the data type of the tensor
	 *
	 * \see setType()
	 */

	virtual DataType getType() const = 0;

	/** \brief whether the tensor is a network input
	 */
	virtual bool isNetworkInput() const = 0;

	/** \brief whether the tensor is a network output
	 */
	virtual bool isNetworkOutput() const = 0;

protected:
	virtual ~ITensor() {}
};

/** \class ILayer
  * 
  * \brief base class for all layer classes in a network definition
  *
  */

class ILayer
{
public:
	/**
	 * \brief return the type of a layer
	 * 
	 * \see LayerType
	 */
	virtual LayerType getType() const = 0;

	/**
	* \brief set the name of a layer
	*
	* this method copies the name string
	* 
	* \see getName()
	*/
	virtual void setName(const char* name) = 0;

	/**
	* \brief return the name of a layer
	*
	* \see setName()
	*/
	virtual const char* getName() const = 0;


	/**
 	 * \brief get the number of inputs of a layer 
	 */
	virtual int getNbInputs() const = 0;

	/**
	 * \brief get the layer input corresponding to the given index
	 *
	 * \param index the index of the input tensor
	 *
	 * \return the input tensor, or nullptr if the index is out of range
	 */

	virtual ITensor* getInput(int index) const = 0;

	/**
	 * \brief get the number of outputs of a layer
	 */
	virtual int getNbOutputs() const = 0;

	/**
	 * \brief get the layer output corresponding to the given index
	 *
	 * \return the indexed output tensor, or nullptr if the index is out of range
	 */
	virtual ITensor* getOutput(int index) const = 0;

protected:
	virtual ~ILayer() {}
};

/** \class IConvolutionLayer
 *
 * \brief a convolution layer in a network definition
 *
 * This layer performs a correlation operation between 3-dimensional filter with a 4-dimensional tensor to produce another 4-dimensional tensor. 
 *
 * The HW output size of the convolution is set according to the \p INetworkCustomDimensions set in INetworkDefinition::setCustomConvolutionDimensions().
 *
 * An optional bias argument is supported, which adds a per-channel constant to each value in the output.
 */

class IConvolutionLayer : public ILayer
{
public:

	/**
	* \brief set the HW kernel size of the convolution
	*
	* \see getKernelSize()
	*/
	virtual void setKernelSize(DimsHW kernelSize) = 0;

	/**
 	 * \brief get the HW kernel size of the convolution
	 *
	 * \see setKernelSize()
	 */
	virtual DimsHW getKernelSize() const = 0;

	/**
	 * \brief set the number of output maps for the convolution
	 *
	 * \see getNbOutputMaps()
	 */
	virtual void setNbOutputMaps(int nbOutputMaps) = 0;

	/**
 	 * \brief get the number of output maps for the convolution
	 *
	 * \see setNbOutputMaps()
	 */
	virtual int getNbOutputMaps() const = 0;

	/**
	 * \brief get the stride of the convolution
	 *
	 * default: (1,1)
	 *
	 * \see setStride()
	 */
	virtual void setStride(DimsHW stride) = 0;

	/**
	 * \brief get the stride of the convolution
	 */
	
	virtual DimsHW getStride() const = 0;

	/**
	 * \brief set the padding of the convolution
	 *
	 * The input will be zero-padded by this number of elements in the height and width directions. Padding is symmetric.
	 *
	 * default: (0,0)
	 *
	 * \see getPadding()
	 */
	virtual void setPadding(DimsHW padding) = 0;

	/**
	* \brief get the padding of the convolution
	*
	* \see setPadding()
	*/
	virtual DimsHW getPadding() const = 0;				// padding defaults to 0

	/**
	 * \brief set the number of groups for a convolution
	 *
	 * The input tensor channels are  divided into \p nbGroups groups, and a convolution is executed for each group, using a filter per group. The results of the group
	 * convolutions are concatenated to form the output.
	 *
	 * \note When using groups in int8 mode, the size of the groups (i.e. the channel count divided by the group count) must be a multiple of 4 for both input and output.
	 *
	 * default: 1
	 *
	 * \see getNbGroups()
	 */

	virtual void setNbGroups(int nbGroups) = 0;

	/**
	 * \brief set the number of groups for a convolution
	 *
	 * \see setNbGroups()
	 */

	virtual int getNbGroups() const = 0; 


	/**
	 * \brief set the kernel weights for the convolution
	 *
	 * The weights are specified as a contiguous array in \p GKCRS order, where \p G is the number of groups, \p K the number of output feature maps, \p C the number of
	 * input channels, and \p R and \p S are the height and width of the filter
	 *
	 * \see getWeights()
	 */
	virtual void setKernelWeights(Weights weights) = 0;

	/**
	 * \brief get the kernel weights for the convolution
	 *
	 * \see setNbGroups()
	 */

	virtual Weights getKernelWeights() const = 0;

	/**
	 * \brief set the bias weights for the convolution
	 *
	 * Bias is optional. To omit bias, set the count value of the weights structure to zero.
	 *
	 * The bias is applied per-channel, so the number of weights (if non-zero) must be equal to the number of output feature maps.
	 *
	 * \see getBiasWeights()
	 */
	virtual void setBiasWeights(Weights weights) = 0;

	/**
	 * \brief get the bias weights for the convolution
	 *
	 * \see getBiasWeights()
	 */
	virtual Weights getBiasWeights() const = 0;

	/**
	 * \brief set the dilation for a convolution
	 *
	 * default (1,1)
	 * \see getDilation
	 */
	virtual void setDilation(DimsHW dims) = 0;

	/**
	 * \brief get the dilation for a convolution
	 *
	 * \see setDilation
	 */
	virtual DimsHW getDilation() const = 0;

protected:
	virtual ~IConvolutionLayer() {}
};

/** \class IFullyConnectedLayer
 *
 * \brief a fully connected layer in a network definition
 *
 * The layer automatically reshapes its input into a \p NxCx1x1 tensor, then applies a matrix multiplication to create an NxKx1x1 output.
 * An optional bias argument is supported.
 * 
 */

class IFullyConnectedLayer : public ILayer
{
public:
	/**
	 * \brief set the number of outputs from the fully connected layer
	 *
	 * \see getNbOutputChannels()
	 */
	virtual void setNbOutputChannels(int nbOutputs) = 0;

	/**
	 * \brief set the number of outputs from the fully connected layer
	 *
	 * \see setNbOutputChannels()
	 */
	virtual int getNbOutputChannels() const = 0;

	/**
	* \brief set the kernel weights. The expected format is an array of KC values, where K is the number of outputs and C is the number of inputs. 
	*
	* \see getKernelWeights()
	*/
	virtual void setKernelWeights(Weights weights) = 0;

	/**
	 * \brief get the kernel weights.
	 *
	 * \see setKernelWeights()
	 */
	virtual Weights getKernelWeights() const = 0;

	/**
	 * \brief set the bias weights
	 *
	 * Bias is optional. To omit bias, set the count value in the weights structure to zero.
	 * 
	 * \see getBiasWeightsWeights()
	 */
	virtual void setBiasWeights(Weights weights) = 0;

	/**
	* \brief get the bias weights
	*
	* \see setBiasWeightsWeights()
	*/
	virtual Weights getBiasWeights() const = 0;

protected:
	virtual ~IFullyConnectedLayer() {}
};


/** \enum ActivationType
 *
 * \brief enumerates the types of activation to perform in an activation layer.
 *
 */
enum class ActivationType : int
{
	kRELU = 0,			//!< rectified linear activation
	kSIGMOID = 1,		//!< sigmoid activation
	kTANH = 2			//!< TanH activation
};
template<> inline int EnumMax<ActivationType>() { return 3; } //!< maximum number of elements in ActivationType enum. \see ActivationType

/** \class IActivationLayer
 *
 * \brief an Activation layer in a network definition
 *
 * this layer applies a per-element activation function to its input.
 *
 * The output has the same shape as the input.
 */
class IActivationLayer : public ILayer
{
public:
	/**
	 * \brief set the type of activation to be performed
	 *
	 * \see getActivationType(), ActivationType
	 */
	virtual void setActivationType(ActivationType type) = 0;

	/**
	* \brief get the type of activation to be performed
	*
	* \see setActivationType(), ActivationType
	*/
	virtual ActivationType getActivationType() const = 0;
protected:
	virtual ~IActivationLayer() {}
};

/** \enum PoolingType
 *
 * \brief the type of pooling to perform in a pooling layer.
 *
 */

enum class PoolingType : int
{
	kMAX = 0,			// Maximum over elements
	kAVERAGE = 1,		// Average over elements. If the tensor is padded, the count includes the padding
	kMAX_AVERAGE_BLEND = 2		// Blending between the max pooling and average pooling: (1-blendFactor)*maxPool + blendFactor*avgPool
};
template<> inline int EnumMax<PoolingType>() { return 3; } //!< maximum number of elements in PoolingType enum. \see PoolingType


/** \class IPoolingLayer
*
* \brief a Pooling layer in a network definition
*
* The layer applies a reduction operation within a window over the input.

* The output size is determined from the input size using the formula set by INetworkDefinition::setCustomPoolingDimensions().
*/

class IPoolingLayer : public ILayer
{
public:
	/**
	 * \brief set the type of activation to be performed
	 *
	 * \see getPoolingType(), PoolingType
	 */

	virtual void setPoolingType(PoolingType type) = 0;

	/**
	 * \brief get the type of activation to be performed
	 *
	 * \see setPoolingType(), PoolingType
	 */
	virtual PoolingType getPoolingType() const = 0;

	/**
	 * \brief set the window size for pooling
	 *
	 * \see getWindowSize()
	 */
	virtual void setWindowSize(DimsHW windowSize) = 0;

	/**
	 * \brief get the window size for pooling
	 *
	 * \see setWindowSize()
	 */
	virtual DimsHW getWindowSize() const = 0;

	/**
	 * \brief set the stride for pooling
	 *
	 * default: 1
	 *
	 * \see getStride()
	 */
	virtual void setStride(DimsHW stride) = 0;

	/**
	 * \brief get the stride for pooling
	 *
	 * \see setStride()
	 */
	virtual DimsHW getStride() const = 0;

	/**
	 * \brief set the padding for pooling
	 *
	 * default: 0
	 *
	 * \see getStride()
	 */
	virtual void setPadding(DimsHW padding) = 0;

	/**
	 * \brief get the padding for pooling
	 *
	 * default: 0
	 *
	 * \see getStride()
	 */

	virtual DimsHW getPadding() const = 0;

	/**
	* \brief set the blending factor for the max_average_blend mode: max_average_blendPool = (1-blendFactor)*maxPool + blendFactor*avgPool
	* blendFactor is a user value in [0,1] with the default value of 0.0
	* This value only applies for the kMAX_AVERAGE_BLEND mode. 
	*
	* \see getBlendFactor()
	*/
	virtual void setBlendFactor(float blendFactor) = 0;

	/**
	* \brief get the blending factor for the max_average_blend mode: max_average_blendPool = (1-blendFactor)*maxPool + blendFactor*avgPool
	* blendFactor is a user value in [0,1] with the default value of 0.0
	* In modes other than kMAX_AVERAGE_BLEND, blendFactor is ignored
	*
	* \see setBlendFactor()
	*/
	virtual float getBlendFactor() const = 0;

	/**
	* \brief set whether average pooling uses as a denominator the overlap area between the window and the unpadded input. 
	* If this is not set, the denominator is the overlap between the pooling window and the padded input.
	*
	* default: true
	*
	* \see getAverageCountExcludesPadding()
	*/
	virtual void setAverageCountExcludesPadding(bool exclusive) = 0;

	/**
	* \brief get whether exclusive pooling uses as a denominator the overlap area betwen the window and the unpadded input. 
	*
	* \see setAverageCountExcludesPadding()
	*/
	virtual bool getAverageCountExcludesPadding() const = 0;
	
protected:
	virtual ~IPoolingLayer() {}
};


/** \class ILRNLayer
 *
 * \brief a LRN layer in a network definition
 *
 * The output size is the same as the input size
 */


class ILRNLayer : public ILayer
{
public:
	/**
	 * \brief set the LRN window size.
	 *
     * The window size must be odd and in the range of [1, 15]
	 * \see setWindowStride()
	 */

	virtual void setWindowSize(int windowSize) = 0;
	/**
	 * \brief get the LRN window size.
	 *
	 * \see getWindowStride()
	 */
	virtual int getWindowSize() const = 0;

	/**
	 * \brief set the LRN alpha value.
	 *
     * The valid range is [-1e20, 1e20].
	 * \see getAlpha()
	 */
	virtual void setAlpha(float alpha) = 0;

	/**
	 * \brief get the LRN alpha value.
	 *
	 * \see setAlpha()
	 */
	virtual float getAlpha() const = 0;

	/**
	 * \brief set the LRN beta value.
	 *
     * The valid range is [0.01, 1e5f].
	 * \see getBeta()
	 */
	virtual void setBeta(float beta) = 0;

	/**
	 * \brief get the LRN beta value.
	 *
	 * \see setBeta()
	 */
	virtual float getBeta() const = 0;

	/**
	 * \brief set the LRN K value.
	 *
     * The valid range is [1e-5, 1e10].
	 * \see getK()
	 */
	virtual void setK(float k) = 0;

	/**
	 * \brief get the LRN K value.
	 *
	 * \see setK()
	 */
	virtual float getK() const = 0;
protected:
	virtual ~ILRNLayer() {}
};

/** \brief controls how scale is applied in a Scale layer
 *
 * \see IScaleLayer
 */

enum class ScaleMode : int
{
	kUNIFORM = 0,		//!< identical coefficients across all elements of the tensor
	kCHANNEL = 1,		//!< per-channel coefficients
	kELEMENTWISE = 2	//!< elementwise coefficients
};
template<> inline int EnumMax<ScaleMode>() { return 3; } //!< maximum number of elements in ScaleMode enum. \see ScaleMode

/** \class IScaleLayer
 *
 * \brief a Scale layer in a network definition
 *
 * this layer applies a per-element computation to its input:
 * 
 * \p output = (\p input* \p scale + \p shift)^ \p power
 *
 * The coefficients can be applied on a per-tensor, per-channel, or per-element basis.
 * 
 *  if the count value in the weights is 0, a default is used. The default shift is 0, and the default power and scale are 1.
 *
 * The output size is the same as the input size.
 *
 * \see ScaleMode
 */

class IScaleLayer : public ILayer
{
public:
	
	/**
	 * \brief set the scale mode.
	 *
	 * \see getMode()
	 */
	virtual void setMode(ScaleMode mode) = 0;

	/**
	 * \brief set the scale mode.
	 *
	 * \see setMode()
	 */
	virtual ScaleMode getMode() const = 0;

	/**
	 * \brief set the shift value.
	 *
	 * \see getShift()
	 */
	virtual void setShift(Weights shift) = 0;

	/**
	 * \brief get the shift value.
	 *
	 * \see setShift()
	 */
	virtual Weights getShift() const = 0;

	/**
	 * \brief set the scale value.
	 *
	 * \see getScale()
	 */
	virtual void setScale(Weights scale) = 0;

	/**
	 * \brief get the scale value.
	 *
	 * \see setScale()
	 */
	virtual Weights getScale() const = 0;

	/**
	 * \brief set the power value.
	 *
	 * \see getPower()
	 */
	virtual void setPower(Weights power) = 0;

	/**
	 * \brief get the power value.
	 *
	 * \see setPower()
	 */
	virtual Weights getPower() const = 0;

protected:
	virtual ~IScaleLayer() {}
};

/** \class ISoftMaxLayer
 *
 * \brief a Softmax layer in a network definition
 * 
 * This layer applies a per-channel softmax to its input
 * 
 * The output size is the same as the input size.
 *
 */

class ISoftMaxLayer : public ILayer
{
protected:
	virtual ~ISoftMaxLayer() {}
};


/** \class IConcatenationLayer
 *
 * \brief a concatenation layer in a network definition
 * 
 * The output size is the sum of all tensors after concatenated
 * across channels.
 *
 */

class IConcatenationLayer : public ILayer
{
protected:
	virtual ~IConcatenationLayer() {}
};


/** \class IDeconvolutionLayer
 *
 * \brief a deconvolution layer in a network definition
 *
 * The output size is defined using the formula set by INetworkDefinition::setDeconvolutionOutputDimensionsFormula()
 *
 */

class IDeconvolutionLayer : public ILayer
{
public:
	/**
	 * \brief set the HW kernel size of the convolution
	 *
	 * \see getKernelSize()
	 */
	virtual void setKernelSize(DimsHW kernelSize) = 0;

	/**
	 * \brief get the HW kernel size of the deconvolution
	 *
	 * \see setKernelSize()
	 */
	virtual DimsHW getKernelSize() const = 0;

	/**
	 * \brief set the number of output feature maps for the deconvolution
	 *
	 * \see getNbOutputMaps()
	 */
	virtual void setNbOutputMaps(int nbOutputMaps) = 0;

	/**
	 * \brief get the number of output feature maps for the deconvolution
	 *
	 * \see setNbOutputMaps()
	 */
	virtual int getNbOutputMaps() const = 0;

	/**
	 * \brief get the stride of the deconvolution
	 *
	 * \see setStride()
	 */
	virtual void setStride(DimsHW stride) = 0;

	/**
	 * \brief get the stride of the deconvolution
	 *
	 * default: (1,1)
	 */
	virtual DimsHW getStride() const = 0;


	/**
	 * \brief set the padding of the deconvolution
	 *
	 * The input will be zero-padded by this number of elements in the height and width directions. Padding is symmetric.
	 *
	 * default: (0,0)
	 *
	 * \see getPadding()
	 */
	virtual void setPadding(DimsHW padding) = 0;

	/**
	 * \brief get the padding of the deconvolution
	 *
	 * \see setPadding()
	 */
	virtual DimsHW getPadding() const = 0;				// padding defaults to 0
	
    /**
	 * \brief set the number of groups for a deconvolution
	 *
	 * The input tensor channels are  divided into \p nbGroups groups, and a deconvolution is executed for each group, using a filter per group. The results of the group
	 * convolutions are concatenated to form the output.
	 *
	 * \note When using groups in int8 mode, the size of the groups (i.e. the channel count divided by the group count) must be a multiple of 4 for both input and output.
	 *
	 * default: 1
	 *
	 * \see getNbGroups()
	 */

	virtual void setNbGroups(int nbGroups) = 0;

	/**
	 * \brief set the number of groups for a deconvolution
	 *
	 * \see setNbGroups()
	 */

	virtual int getNbGroups() const = 0; 

	/**
	 * \brief set the kernel weights for the deconvolution
	 *
	 * The weights are specified as a contiguous array in \p CKRS order, where \p C the number of
	 * input channels, \p K the number of output feature maps, and \p R and \p S are the height and width of the filter
	 *
	 * @ see getWeights()
	 */
	virtual void setKernelWeights(Weights weights) = 0;

	/**
	 * \brief get the kernel weights for the deconvolution
	 *
	 * \see setNbGroups()
	 */

	virtual Weights getKernelWeights() const = 0;

	/**
	 * \brief set the bias weights for the deconvolution
	 *
	 * Bias is optional. To omit bias, set the count value of the weights structure to zero.
	 *
	 * The bias is applied per-feature-map, so the number of weights (if non-zero) must be equal to the number of output feature maps.
	 *
	 * \see getBiasWeights()
	 */
	virtual void setBiasWeights(Weights weights) = 0;

	/**
	 * \brief get the bias weights for the deconvolution
	 *
	 * \see getBiasWeights()
	 */
	virtual Weights getBiasWeights() const = 0;


protected:
	virtual ~IDeconvolutionLayer() {}
};

/**
 * \enum ElementWiseOperation
 *
 * \brief enumerates the binary operations that may be performed by an ElementWise layer
 *
 * \see IElementWiseLayer
 */
enum class ElementWiseOperation : int
{
	kSUM = 0,		//!< sum of the two elements
	kPROD = 1,		//!< product of the two elements
	kMAX = 2,		//!< maximum of the two elements
	kMIN = 3,		//!< minimum of the two elements
	kSUB = 4,		//!< substract the second element from the first
	kDIV = 5,		//!< divide the first element by the second
	kPOW = 6		//!< the first element to the power of the second element

};
template<> inline int EnumMax<ElementWiseOperation>() { return 7; } //!< maximum number of elements in ElementWiseOperation enum. \see ElementWiseOperation

/** \class IElementWiseLayer
 *
 * \brief a elementwise layer in a network definition
 * 
 * This layer applies a per-element binary operation between corresponding elements of two tensors.
 *
 * The input dimensions of the two input tensors must be equal, and the output tensor is the same size as each input.
 *
 */

class IElementWiseLayer : public ILayer
{
public:
	/**
	 * \brief set the binary operation for the layer
	 *
	 * \see getOperation(), ElementWiseOperation
	 *
	 * \see getBiasWeights()
	 */
	virtual void setOperation(ElementWiseOperation type) = 0;

	/**
	 * \brief get the binary operation for the layer
	 *
	 * \see setOperation(), ElementWiseOperation
	 *
	 * \see setBiasWeights()
	 */
	virtual ElementWiseOperation getOperation() const = 0;
protected:
	virtual ~IElementWiseLayer() {}
};

/**
 * \enum RNNOperation
 *
 * \brief enumerates the RNN operations that may be performed by an RNN layer
 *
 * Equation definitions:
 * i - input gate
 * o - output gate
 * f - forget gate
 * z - update gate
 * r - reset gate
 * c - cell gate
 * h - hidden gate
 * t - time step(t-1 means previous time step)
 * Xi - input tensor
 * W[izrfcoh] - W parameter weight matrices for the corresponding gates
 * R[izrfcoh] - U parameter weight matrices for the corresponding gates
 * Wb[izrfcoh] - W parameter bias vectors for the corresponding gates
 * Rb[izrfcoh] - U parameter bias vectors for the corresponding gates
 * ReLU(X) - max(X, 0)
 * tanh - hyperbolic tangent of X
 * sigmoid(X) - 1 / (1 + e^-X)
 * [C|H] - Cell/Hidden state
 *
 * - Equations:
 *   kRELU
 *   - Ht = ReLU(WiXt + RiHt-1 + Wbi + Rbi)
 *   kTANH
 *   - Ht = TANH(WiXt + RiHt-1 + Wbi + Rbi)
 *   kLSTM
 *   - it = sigmoid(WiXt + RiHt-1 + Wbi + Rbi)
 *   - ft = sigmoid(WfXt + RfHt-1 + Wbf + Rbf)
 *   - ot = sigmoid(WoXt + RoHt-1 + Wbo + Rbo)
 *   - ct = tanh(WcXt + RcHt-1 + Wbc + Rbc)
 *   - C = f * Ct-1 + it * ct
 *   - H = ot * tanh(C)
 *   kGRU
 *   - zt = sigmoid(WzXt + RzHt-1 + Wbz + Rbz)
 *   - rt = sigmoid(WrXt + RrHt-1 + Wbr + Rbr)
 *   - ht = tanh(WhXt + rt *(RhHt-1 + Rbh) + Wbh)
 *   - H = (1 - zt) * ht + it * Ht-1
 *
 * \see IRNNLayer
 */
enum class RNNOperation : int
{
    kRELU = 0, //!< Single gate RNN w/ ReLU activation function.
    kTANH = 1, //!< Single gate RNN w/ TANH activation function.
    kLSTM = 2, //!< Four-gate LSTM network w/o peephole connections.
    kGRU  = 3  //!< Three-gate network consisting of Gated Recurrent Units.
};
template <> inline int EnumMax<RNNOperation>() { return 4; } //!< maximum number of elements in RNNOperation enum. \see RNNOperation

/**
 * \enum RNNDirection
 *
 * \brief enumerates the RNN direction that may be performed by an RNN layer
 *
 * \see IRNNLayer
 */
enum class RNNDirection : int
{
    kUNIDIRECTION = 0, //!< Network iterations from first input to last input.
    kBIDIRECTION  = 1  //!< Network iterates from first to last and vice versa and outputs concatenated.
}; 
template <> inline int EnumMax<RNNDirection>() { return 2; } //!< maximum number of elements in RNNDirection enum. \see RNNDirection

/**
 * \enum RNNInputMode
 *
 * \brief enumerates the RNN input modes that may occur with an RNN layer
 *
 * \see IRNNLayer
 */
enum class RNNInputMode : int
{
    kLINEAR = 0, //!< Perform the normal matrix multiplication in the first recurrent layer.
    kSKIP   = 1  //!< No operation is performed on the first recurrent layer.
};

template <> inline int EnumMax<RNNInputMode>() { return 2; } //!< maximum number of elements in RNNInputMode enum. \see RNNInputMode

/**
 * \class IRNNLayer
 *
 * \brief a RNN layer in a network definition
 *
 * This layer applies an RNN operation on the inputs.
 */

class IRNNLayer : public ILayer
{
    public:
        /**
         * \brief get the number of layers in the RNN.
         *
         * \return The number of layers in the RNN.
         */
        virtual unsigned getLayerCount() const = 0;

        /**
         * \brief get the size of the hidden layers.
         *
         * The hidden size is the value of hiddenSize parameter passed into addRNN().
         *
         * \return The internal hidden layer size for the RNN.
         * \see getDirection(), addRNN()
         */
        virtual std::size_t getHiddenSize() const = 0;

        /**
         * \brief get the sequence length
         *
         * The sequence length is the maximum number of time steps passed into the addRNN() function.
         * This is also the maximum number of input tensors that the RNN can process at once.
         *
         * \return the maximum number of time steps that can be executed by a single call RNN layer.
         */
        virtual int getSeqLength() const = 0;

        /**
         * \brief set the operation of the RNN layer.
         *
         * \see getOperation(), RNNOperation
         */
        virtual void setOperation(RNNOperation op) = 0;

        /**
         * \brief get the operation of the RNN layer.
         *
         * \see setOperation(), RNNOperation
         */
        virtual RNNOperation getOperation() const = 0;

        /**
         * \brief set the operation of the RNN layer.
         *
         * \see getInputMode(), RNNInputMode
         */
        virtual void setInputMode(RNNInputMode op) = 0;

        /**
         * \brief get the operation of the RNN layer.
         *
         * \see setInputMode(), RNNInputMode
         */
        virtual RNNInputMode getInputMode() const = 0;
        
        /**
         * \brief set the direction of the RNN layer.
         *
         * The direction determines if the RNN is run
         * as a unidirectional(left to right) or
         * bidirectional(left to right and right to left).
         * In the #RNNDirection::kBIDIRECTION case the
         * output is concatenated together, resulting
         * in output size of 2x getHiddenSize().
         * \see getDirection(), RNNDirection
         */
        virtual void setDirection(RNNDirection op) = 0;

        /**
         * \brief get the direction of the RNN layer.
         *
         * \see setDirection(), RNNDirection
         */
        virtual RNNDirection getDirection() const = 0;

        /**
         * \param weights The weight structure holding the weight parameters.
         *
         * \brief set the weight parameters for the RNN.
         *
         * The trained weights for the weight parameter matrix of the RNN.
         * The data type must be of the type #DataType::kFLOAT or #DataType::kHALF.
         * The weight structure holds two sets of parameters, one for W and one for R. \see #RNNOperation
         * Each parameter matrix is linearly appended after the previous parameter matrix without padding.
         * The format of the Weight matrix is {L, N, C} defined as:
         *  - L - The number of layers in the RNN, equal to getLayerCount()
         *  - N - The number of gates matrices in the RNN, dependent on getOperation().
         *  -- If getOperation() is #RNNOperation::kRELU or #RNNOperation::kTANH there are 2 gate matrices, with order Winput, Uinput).
         *  -- If getOperation() is #RNNOperation::kLSTM there are 8 gate matrices, with order Wforget, Winput, Wcell, Woutput, Uforgot, Uinput, Ucell, Uoutput.
         *  -- If getOperation() is #RNNOperation::kGRU there are 6 gate matrices, with order Wupdate, Wreset, Whidden, Uupdate, Ureset, Uhidden.
         *  - C - The size of each weight matrix, which varies.
         *  -- If the mode is #RNNInputMode::kLINEAR and #RNNDirection::kUNIDIRECTION then for first layer:
         *  --- Each sub-matrix consists of {getHiddenSize(),     getDataLength()}
         *  -- If the mode is #RNNInputMode::kLINEAR and #RNNDirection::kBIDIRECTION then for first layer:
         *  --- Each sub-matrix consists of {getHiddenSize(),     getDataLength() + getHiddenSize()}
         *  -- Otherwise all other layers have the dimensions:
         *  --- Each sub-matrix consists of {getHiddenSize(),     getHiddenSize()} elements if getDirection() is #RNNDirection::kUNIDIRECTION.
         *  --- Each sub-matrix consists of {getHiddenSize(), 2 x getHiddenSize()} elements if getDirection() is #RNNDirection::kBIDIRECTION.
         *
         * \see getWeights(), #RNNOperation
         */
        virtual void setWeights(Weights weights) = 0;

        /**
         * \brief get the W weights for the RNN
         *
         * \see setWeights()
         */
        virtual Weights getWeights() const = 0;

        /**
         * \param bias The weight structure holding the bias parameters.
         *
         * \brief set the weight parameters for the RNN.
         *
         * The trained weights for the bias parameter vector of the RNN.
         * The data type must be of the type #DataType::kFLOAT or #DataType::kHALF.
         * The weight structure holds two sets of parameters, one for W and one for R. \see #RNNOperation
         * Each parameter vector is linearly appended after the previous parameter matrix without padding.
         * The format of the Bias vector is {L, N, C} defined as:
         *  - L - The number of layers in the RNN, equal to getLayerCount()
         *  - N - The number of gates vectors in the RNN, dependent on getOperation().
         *  -- If getOperation() is #RNNOperation::kRELU or #RNNOperation::kTANH there are 2 gate vectors, with order Winput, Uinput).
         *  -- If getOperation() is #RNNOperation::kLSTM there are 8 gate vectors, with order Wforget, Winput, Wcell, Woutput, Uforgot, Uinput, Ucell, Uoutput.
         *  -- If getOperation() is #RNNOperation::kGRU there are 6 gate vectors, with order Wupdate, Wreset, Whidden, Uupdate, Ureset, Uhidden.
         *  - C - The size of each bias vector, which varies.
         *  --- Each sub-vector consists of {getHiddenSize(),     getHiddenSize()} elements if getDirection() is #RNNDirection::kUNIDIRECTION.
         *  --- Each sub-vector consists of {getHiddenSize(), 2 x getHiddenSize()} elements if getDirection() is #RNNDirection::kBIDIRECTION.
         *
         * \see getBias(), #RNNOperation
         */
       
        virtual void setBias(Weights bias) = 0;

        /**
         * \brief get the bias parameter vector for the RNN
         *
         * \see setB()
         */
        virtual Weights getBias() const = 0;

        /**
         * \brief get the length of the data being processed by the RNN for use in computing
         * other values.
         *
         * \see setHiddenState(), setCellState()
         */
        virtual int getDataLength() const = 0;

        /**
         * \param hidden The initial hidden state of the RNN.
         *
         * \brief Set the initial hidden state of the RNN with the provided \p hidden ITensor.
         *
         * The layout for \p hidden is a linear layout of a 3D matrix:
         *  - C - The number of layers in the RNN, it must match getLayerCount().
         *  - H - The number of mini-batches for each time sequence.
         *  - W - The size of the per layer hidden states, it must match getHiddenSize().
         *
         * The amount of space required is doubled if getDirection() is #RNNDirection::kBIDIRECTION with the bidirectional states coming after the unidirectional states.
         *
         * If hidden is not specified, then the initial hidden state is set to zero.
         *
         * \see getHiddenState()
         */
        virtual void setHiddenState(ITensor &hidden) = 0;

        /**
         * \brief Get the initial hidden state of the RNN.
         *
         * \return nullptr if no initial hidden tensor was specified, the initial hidden data otherwise.
         */
        virtual ITensor *getHiddenState() const = 0;

        /**
         * \param cell The initial cell state of the RNN.
         *
         * \brief Set the initial cell state of the RNN with the provided \p cell ITensor.
         *
         * The layout for \p cell is a linear layout of a 3D matrix:
         *  - C - The number of layers in the RNN, it must match getLayerCount().
         *  - H - The number of mini-batches for each time sequence.
         *  - W - The size of the per layer hidden states, it must match getHiddenSize().
         *
         * If \p cell is not specified, then the initial cell state is set to zero.
         *
         * The amount of space required is doubled if getDirection() is #RNNDirection::kBIDIRECTION with the bidirectional states coming after the unidirectional states.
         *
         * The cell state only affects LSTM RNN's.
         *
         * \see getCellState()
         */
        virtual void setCellState(ITensor &cell) = 0;

        /**
         * \brief Get the initial cell state of the RNN.
         *
         * \return nullptr if no initial cell tensor was specified, the initial cell data otherwise.
         */
        virtual ITensor* getCellState() const = 0;

    protected:
        virtual ~IRNNLayer() {}
};

/** \class IOutputDimensionsFormula
 *
 * \brief application-implemented inteface to compute layer output sizes
 *
 */

class IOutputDimensionsFormula
{
public:
	/** \brief application-implemented interface to compute the HW output dimensions of a layer from the layer input and parameters
	 * 
	 * \param inputDims the input dimensions of the layer
	 * \param kernelSize the kernel size (or window size, for a pooling layer) parameter of the layer operation
	 * \param stride the stride parameter for the layer
	 * \param padding the padding parameter of the layer
	 * \param dilation the dilation parameter of the layer (only applicable to convolutions)
	 * \param layerName the name of the layer
	 *
	 * \return the output size of the layer
	 *
	 * note that for dilated convolutions, the dilation is applied to the kernel size before this routine is called
	 */
	virtual DimsHW compute(DimsHW inputDims, DimsHW kernelSize, DimsHW stride, DimsHW padding, DimsHW dilation, const char* layerName) = 0;
protected:
	~IOutputDimensionsFormula() {}
};


/** \class IPlugin
*
* \brief plugin class for user-implemented layers
*
* plugins are a mechanism for applications to implement custom layers. Each plugin is owned by the application, and its lifetime
* must span any use of it by TensorRT
*
*/

class IPlugin
{
public:
	/**
	* \brief get the number of outputs from the layer
	*
	* \return the number of outputs
	*
	* this function is called by the implementations of INetworkDefinition and IBuilder. In particular, it is called prior to any call to initialize().
	*/

	virtual int getNbOutputs() const = 0;
	/**
	* \brief get the dimension of an output tensor
	*
	* \param index the index of the output tensor
	* \param inputs the input tensors
	* \param nbInputDims the number of input tensors
	*
	* this function is called by the implementations of INetworkDefinition and IBuilder. In particular, it is called prior to any call to initialize().
	*/
	virtual Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) = 0;

	/**
	* \brief configure the layer
	*
	* this function is called by the builder prior to initialize(). It provides an opportunity for the layer to make algorithm choices on the basis
	* of its weights, dimensions, and maximum batch size
	*
	* \param inputDims the input tensor dimensions
	* \param nbInputs the number of inputs
	* \param outputDims the output tensor dimensions
	* \param nbOutputs the number of outputs
	* \param maxBatchSize the maximum batch size
	*
	* the dimensions passed here do not include the outermost batch size (i.e. for 2-D image networks, they will be 3-dimensional CHW dimensions)
	*/
	virtual void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) = 0;

	/**
	* \brief initialize the layer for execution. This is called when the engine is created.
	*
	*
	* \return 0 for success, else non-zero (which will cause engine termination.)
	*
	*/
	virtual int initialize() = 0;

	/**
	* \brief shutdown the layer. This is called when the engine is destroyed
	*/
	virtual void terminate() = 0;


	/**
	* \brief find the workspace size required by the layer
	*
	* this function is called during engine startup, after initialize(). The workspace size returned should be sufficient for any
	* batch size up to the maximum
	*
	* \return the workspace size
	*/
	virtual size_t getWorkspaceSize(int maxBatchSize) const = 0;


	/**
	* \brief execute the layer
	*
	* \param batchSize the number of inputs in the batch
	* \param inputs the memory for the input tensors
	* \param outputs the memory for the output tensors
	* \param workspace workspace for execution
	* \param stream the stream in which to execute the kernels
	*
	* \return 0 for success, else non-zero (which will cause engine termination.)
	*/
	virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) = 0;

	/**
	* \brief find the size of the serialization buffer required
	*
	* \return the size of the serialization buffer
	*/

	virtual size_t getSerializationSize() = 0;

	/**
	* \brief serialize the layer
	*
	* \param buffer a pointer to a buffer of size at least that returned by getSerializationSize()
	*
	* \see getSerializationSize()
	*/
	virtual void serialize(void* buffer) = 0;
protected:
	virtual ~IPlugin() {}
};

/** \class IPluginLayer
 *
 * \brief layer type for plugins
 *
 * \see IPlugin
 */

class IPluginLayer : public ILayer
{
public:
	/**
	* \brief get the plugin for the layer
	*
	* \see IPlugin
	*/
	virtual IPlugin& getPlugin() = 0;

protected:
	virtual ~IPluginLayer() {}
};


/**
* \enum UnaryOperation
*
* \brief enumerates the binary operations that may be performed by an ElementWise layer
*
* \see IElementWiseLayer
*/
enum class UnaryOperation : int
{
	kEXP = 0,		//!< exponentiation
	kLOG = 1,		//!< log (base e)
	kSQRT = 2,		//!< square root
	kRECIP = 3,		//!< reciprocal
	kABS = 4,		//!< absolute value
	kNEG = 5,		//!< negation
};
template<> inline int EnumMax<UnaryOperation>() { return 6; } //!< maximum number of elements in ElementWiseOperation enum. \see ElementWiseOperation


/** \class IUnaryLayer
*
* \brief layer that represents a unary operation
*/

class IUnaryLayer : public ILayer
{
public:
	/**
	* \brief set the binary operation for the layer
	*
	* \see getOperation(), UnaryOperation
	*
	* \see getBiasWeights()
	*/
	virtual void setOperation(UnaryOperation op) = 0;

	/**
	* \brief get the binary operation for the layer
	*
	* \see setOperation(), UnaryOperation
	*
	* \see setBiasWeights()
	*/
	virtual UnaryOperation getOperation() const = 0;
protected:
	virtual ~IUnaryLayer() {}
};

/** \class INetworkDefinition
*
* \brief a network definition for input to the builder
*
*/

/** \class IPaddingLayer
*
* \brief layer that represents a padding operation
*/

class IPaddingLayer : public ILayer
{
public:
	/**
	* \brief set the padding that is applied at the start of the tensor
	*
	* Negative padding results in trimming the edge by the specified amount
	*
	* \see getPrePadding
	*/
	virtual void setPrePadding(DimsHW padding) = 0;

	/**
	* \brief set the padding that is applied at the start of the tensor
	*
	* \see setPrePadding
	*/
	virtual DimsHW getPrePadding() const = 0;

	/**
	* \brief set the padding that is applied at the end of the tensor
	*
	* Negative padding results in trimming the edge by the specified amount
	*
	* \see getPostPadding
	*/
	virtual void setPostPadding(DimsHW padding) = 0;

	/**
	* \brief set the padding that is applied at the end of the tensor
	*
	* \see setPostPadding
	*/
	virtual DimsHW getPostPadding() const = 0;

protected:
	virtual ~IPaddingLayer() {}
};


/** \class IShuffleLayer
*
* \brief layer type for shuffling data
* 
* this class shuffles data by applying applying in sequence: a transpose operation, a reshape operation
* and a second transpose operation. The dimension types of the output are those of the reshape dimension.
*/

struct Permutation
{
	/**
	 * the elements of the permutation.
	 * The permutation is applied as outputDimension = permutation.order[inputDimension], so to
	 * permute from CHW order to HWC order, the required permutation is [1, 2, 0], and to permute
	 * from HWC to CHW, the required permutation is [2, 0, 1].
	 */

	int order[Dims::MAX_DIMS];
};

class IShuffleLayer : public ILayer
{
public:
	/**
	* \brief set the permutation applied by the first transpose operation
	*
	* \param permutation the dimension permutation applied before the reshape
	* 
	* the default is the identity permutation.
	*
	* @see getFirstTranspose
	*/
	virtual void setFirstTranspose(Permutation permutation) = 0;

	/**
	* \brief get the permutation applied by the first transpose operation 
	*
	* \return the dimension permutation applied before the reshape
	*
	* @see setFirstTranspose
	*/
	virtual Permutation getFirstTranspose() const = 0;

	/**
	* \brief set the reshaped dimensions
	*
	* \param dimensions the reshaped dimensions
	*
	* When specifying the new dimensions, 0 means copy from the input, and -1 means
	* infer the dimension from the input and the other dimensions.
	* 
	* The product of the new dimensions must be equal to the product of the old. 
	*/
	virtual void setReshapeDimensions(Dims dimensions) = 0;

	/**
	* \brief get the reshaped dimensions
	*
	* \return the reshaped dimensions
	*/
	virtual Dims getReshapeDimensions() const = 0;

	/**
	* \brief set the permutation applied by the second transpose operation
	*
	* \param permutation the dimension permutation applied after the reshape
	* 
	* the default is the identity permutation.
	*
	* The permutation is applied as outputDimension = permutation.order[inputDimension], so to
	* permute from CHW order to HWC order, the required permutation is [1, 0, 2]
	*
	* @see getSecondTranspose
	*/
	virtual void setSecondTranspose(Permutation permutation) = 0;

	/**
	* \brief get the permutation applied by the second transpose operation 
	*
	* \return the dimension permutation applied after the reshape
	*
	* @see setSecondTranspose
	*/
	virtual Permutation getSecondTranspose() const = 0;

protected:
	virtual ~IShuffleLayer() {}
};

/** \class INetworkDefinition
*
* \brief a network definition for input to the builder
*
*/

class INetworkDefinition
{
public:
	/** \brief add an input tensor to the network
	 *
	 * the name of the input tensor is used to find the index into the buffer array for an engine built from the network
	 * 
	 * \param name the name of the tensor
	 * \param type the type of the data held in the tensor
	 * \param dimensions the dimensions of the tensor
     *
     * Only DataType::kFLOAT and DataType::kHALF are valid input tensor types.
     * The volume of the dimension, including the maximum batch size, must be less than 2^30 elements.
	 *
     * \see ITensor
     *
	 * \return the new tensor or nullptr if there is an error
	 */
	virtual ITensor* addInput(const char* name, DataType type, Dims dimensions) = 0;

	/** \brief mark a tensor as a network output
	 *
	 * \param tensor the tensor to mark as an output tensor
	 */

	virtual void markOutput(ITensor& tensor) = 0;

	/** \brief add a convolution layer to the network
	 *
	 * \param input the input tensor to the convolution
	 * \param nbOutputMaps the number of output feature maps for the convolution
	 * \param kernelSize the HW-dimensions of the convolution kernel
	 * \param kernelWeights the kernel weights for the convolution
	 * \param biasWeights the optional bias weights for the convolution
	 *
	 * \see IConvolutionLayer
	 *
	 * \return the new convolution layer, or null if it could not be created
	 */

	virtual IConvolutionLayer*			addConvolution(ITensor& input, int nbOutputMaps, DimsHW kernelSize, Weights kernelWeights, Weights biasWeights) = 0;

	/** \brief add a fully connected layer to the network
	 *
	 * \param input the input tensor to the layer
	 * \param nbOutputs the number of outputs of the layer
	 * \param kernelWeights the kernel weights for the convolution
	 * \param biasWeights the optional bias weights for the convolution
	 *
	 * \see IFullyConnectedLayer
	 *
	 * the input to a fully connected layer is automatically flattened to a tensor of the form \p NxCx1x1, and the output is of the
	 * form \p NxKx1x1, where C is the nunber of input activations per image, and K is the number of outputs per image.
	 *
	 * \return the new convolution layer, or null if it could not be created
	 */

	virtual IFullyConnectedLayer*		addFullyConnected(ITensor& input, int nbOutputs, Weights kernelWeights, Weights biasWeights) = 0;

	/** \brief add an activation layer to the network
	 *
	 * \param input the input tensor to the layer
	 * \param type the type of activation function to apply
	 *
	 * \see IActivationLayer ActivationType
	 *
	 * \return the new convolution layer, or null if it could not be created
	 */

	virtual IActivationLayer*			addActivation(ITensor& input, ActivationType type) = 0;

	/** \brief add a pooling layer to the network
	 *
	 * \param input the input tensor to the layer
	 * \param type the type of pooling to apply
	 * \param windowSize the size of the pooling window
	 *
	 * \see IPoolingLayer PoolingType
	 *
	 * \return the new pooling layer, or null if it could not be created
	 */

	virtual IPoolingLayer*				addPooling(ITensor& input, PoolingType type, DimsHW windowSize) = 0;

	/** \brief add a LRN layer to the network
	 * 
	 * \param input the input tensor to the layer
	 * \param window the size of the window
	 * \param alpha the alpha value for the LRN computation
	 * \param beta the beta value for the LRN computation
	 * \param k the k value for the LRN computation
	 *
	 * \see ILRNLayer 
	 *
	 * \return the new LRN layer, or null if it could not be created
	 */

	virtual ILRNLayer*					addLRN(ITensor& input, int window, float alpha, float beta, float k) = 0;

	/** \brief add a Scale layer to the network
	 *
	 * \param input the input tensor to the layer
	 * \param mode the scaling mode
	 * \param shift the shift value
	 * \param scale the scale value
	 * \param power the power value
     *
     * If the weights are available, then the size of weights are dependent on the on the ScaleMode.
     * For #ScaleMode::kUNIFORM, the number of weights is equal to 1.
     * For #ScaleMode::kCHANNEL, the number of weights is equal to the channel dimension.
     * For #ScaleMode::kELEMENTWISE, the number of weights is equal to the volume of the input.
	 *
	 * \see IScaleLayer
	 *
	 * \return the new Scale layer, or null if it could not be created
	 */

	virtual IScaleLayer*			    addScale(ITensor& input, ScaleMode mode, Weights shift, Weights scale, Weights power) = 0;

	/** \brief add a Scale layer to the network
	 *
	 * \see ISoftMaxLayer
	 *
	 * \return the new SoftMax layer, or null if it could not be created
	 */


	virtual ISoftMaxLayer*				addSoftMax(ITensor& input) = 0;

	/** \brief add a concatenation layer to the network
	*
	* \param inputs the input tensors to the layer
	* \param nbInputs the number of input tensors
	*
    * \see IConcatenationLayer
    *
	* \return the new concatenation layer, or null if it could not be created
    *
    * \warning All tensors must have the same dimensions for all dimensions except for channel.
	*/

	virtual IConcatenationLayer*		addConcatenation(ITensor*const * inputs, int nbInputs) = 0;

	/** \brief add a deconvolution layer to the network
	 *
	 * \param input the input tensor to the layer
	 * \param nbOutputMaps the number of output feature maps
	 * \param kernelSize the HW-dimensions of the convolution kernel
	 * \param kernelWeights the kernel weights for the convolution
	 * \param biasWeights the optional bias weights for the convolution
	 *
     * \see IDeconvolutionLayer
     *
	 * \return the new deconvolution layer, or null if it could not be created
	 */

	virtual IDeconvolutionLayer*		addDeconvolution(ITensor& input, int nbOutputMaps, DimsHW kernelSize, Weights kernelWeights, Weights biasWeights) = 0;

	/** \brief add an elementwise layer to the network
	 *
	 * \param input1 the first input tensor to the layer
	 * \param input2 the second input tensor to the layer
	 * \param op the binary operation that the layer applies
	 *
     * \see IElementWiseLayer
     *
	 * \return the new elementwise layer, or null if it could not be created
	 */

	virtual IElementWiseLayer*			addElementWise(ITensor& input1, ITensor& input2, ElementWiseOperation op) = 0;

    /**
     * \param inputs the input tensor to the layer.
     * \param layerCount the number of layers in the RNN.
     * \param hiddenSize the size of the internal hidden state for each layer.
     * \param maxSeqLen the maximum length of the time sequence.
     * \param op the type of RNN to execute.
     * \param mode the input mode for the RNN.
     * \param dir the direction to run the RNN.
     * \param weights the weights for the weight matrix parameters of the RNN.
     * \param bias the weights for the bias vectors parameters of the RNN.
     *
     * \brief add an \p layerCount deep RNN layer to the network with a sequence length of \p maxSeqLen and \p hiddenSize internal state per layer.
     *
     * The layout for the \p input tensor is {1, T, N, C} and defined as follows:
     *  - T - The number of time sequences to be executed.
     *  - N - The number of mini-batches for each time sequence.
     *  - C - The size of data to be submitted to the RNN.
     *
     * The input tensors must be of the type DataType::kFLOAT or DataType::kHALF.
     * The layout of the weights is row major and must be the same datatype as the input tensor.
     * \p weights contain 8 matrices and \p bias contains 8 vectors.
     *
     * 
     * The RNN layer outputs up to three tensors.
     * The first tensor has dim {1, T, N, C}, is the output of the RNN for each timestep, and defined as follows:
     *  - T - The number of sequences to be executed.
     *  - N - The number of mini-batches for each time sequence.
     *  - C - The hidden state for each layer. Equal to getHiddenSize() if getDirection is #RNNDirection::kUNIDIRECTION, and 2x getHiddenSize() otherwise.
     *
     * The second tensor has dimension {1, L, N, H}, is the final hidden state of the RNN, and defined as follows.
     *  - L - The number of layers in the RNN, equal to getLayerCount()
     *  - N - The number of mini-batches for each time sequence.
     *  - H - The hidden state for each layer. Equal to getHiddenSize() if getDirection is #RNNDirection::kUNIDIRECTION, and 2x getHiddenSize() otherwise.
     *
     * The third tensor has dimension {1, L, N, H}, is the final cell state of the RNN, and defined as follows.
     *  - L - The number of layers in the RNN, equal to getLayerCount()
     *  - N - The number of mini-batches for each time sequence.
     *  - H - The hidden state for each layer. Equal to getHiddenSize() if getDirection is #RNNDirection::kUNIDIRECTION, and 2x getHiddenSize() otherwise.
     *
     *  The third tensor is only available if getOperation() is #RNNDirection::kLSTM.
     *
     * \return the new RNN layer, or null if it could not be created.
     * \see IRNNLayer::setW(), IRNNLayer::setU(), IRNNLayer::setB()
     * \see IRNNLayer
     */
    virtual IRNNLayer*                  addRNN(ITensor &inputs, int layerCount, std::size_t hiddenSize, int maxSeqLen, RNNOperation op, RNNInputMode mode, RNNDirection dir, Weights weights, Weights bias) = 0;

	 /** \brief add a plugin layer to the network
	 *
	 * \param inputs the input tensors to the layer
	 * \param nbInputs the number of input tensors
	 * \param plugin the layer plugin
	 *
     * \see IPluginLayer
     *
	 * \return the new plugin layer, or null if it could not be created
	 */

	virtual IPluginLayer*				addPlugin(ITensor*const* inputs, int nbInputs, IPlugin& plugin) = 0;

	/** \brief Add a unary layer to the network
	*
	* \param input the input tensor to the layer
	* \param operation the operation to apply
	*
	* \see IUnaryLayer
	*
	* \return the new unary layer, or null if it could not be created
	*/

	virtual IUnaryLayer*				addUnary(ITensor& input, UnaryOperation operation) = 0;

	/** \brief Add a padding layer to the network
	*
	* \param input the input tensor to the layer
	* \param prePadding the padding to apply to the start of the tensor
	* \param postPadding the padding to apply to the end of the tensor
	*
	* \see IPaddingLayer
	*
	* \return the new padding layer, or null if it could not be created
	*/

	virtual IPaddingLayer*				addPadding(ITensor& input, DimsHW prePadding, DimsHW postPadding) = 0;


	/** \brief add a shuffle layer to the network
	*
	* \param input the input tensor to the layer
	*
	* \return the new shuffle layer, or null if it could not be created
	*/

	virtual IShuffleLayer*				addShuffle(ITensor& input) = 0;

    /** \brief set the pooling output dimensions formula
     *
     * \param formula the formula from computing the pooling output dimensions. If null is passed, the default formula is used.
     *
     * the default formula in each dimension is (inputDim + padding * 2 - kernelSize) / stride + 1
     *
     * \see IOutputDimensionsFormula getPoolingOutputDimensionsFormula()
     */

	virtual void						setPoolingOutputDimensionsFormula(IOutputDimensionsFormula* formula) = 0;

	/** \brief get the pooling output dimensions formula
	 *
	 * \return the formula from computing the pooling output dimensions
	 *
	 * \see IOutputDimensionsFormula setPoolingOutputDimensionsFormula()
	 */
	virtual IOutputDimensionsFormula&	getPoolingOutputDimensionsFormula() const = 0;

	/** \brief set the convolution output dimensions formula
	 *
	 * \deprecated this method does not currently work reliably and will be removed in a future release
     *
	 * \param formula the formula from computing the convolution output dimensions. If null is passed, the default formula is used.
	 *
	 * the default formula in each dimension is (inputDim + padding * 2 - kernelSize) / stride + 1
	 *
	 * \see IOutputDimensionsFormula getConvolutionOutputDimensionsFormula()
	 */
	virtual void						setConvolutionOutputDimensionsFormula(IOutputDimensionsFormula* formula) = 0;

	/** \brief get the convolution output dimensions formula
	 *
	 * \deprecated this method does not currently work reliably and will be removed in a future release
     *
	 * \return the formula from computing the convolution output dimensions
	 *
	 * \see IOutputDimensionsFormula setConvolutionOutputDimensionsFormula()
	 */

	virtual IOutputDimensionsFormula&	getConvolutionOutputDimensionsFormula() const = 0;

	/** \brief set the deconvolution output dimensions formula
	 *
	 * \deprecated this method does not currently work reliably and will be removed in a future release
	 * 
	 * \param formula the formula from computing the deconvolution output dimensions. If null is passed, the default formula is used.
	 *
	 * the default formula in each dimension is (inputDim - 1) * stride + kernelSize - 2 * padding
	 *
	 * \see IOutputDimensionsFormula getDevonvolutionOutputDimensionsFormula()
	 */

	virtual void						setDeconvolutionOutputDimensionsFormula(IOutputDimensionsFormula* formula) = 0;

	/** \brief get the deconvolution output dimensions formula
 	 *
	 * \return the formula from computing the deconvolution output dimensions.
	 *
	 * \deprecated this method does not currently work reliably and will be removed in a future release
	 * 
	 * \see IOutputDimensionsFormula setDeconvolutionOutputDimensionsFormula()
	 */
	virtual IOutputDimensionsFormula&	getDeconvolutionOutputDimensionsFormula() const = 0;

	/** \brief get the number of layers in the network
	 *
	 * \return the number of layers in the network
	 *
	 * \see getLayer()
	 */

	virtual int							getNbLayers() const = 0;

	/** \brief get the layer specified by the given index
	 *
	 * \param index the index of the layer
	 *
	 * \return the layer, or null if the index is out of range
	 *
	 * \see getNbLayers()
	 */
	virtual ILayer*						getLayer(int index) const = 0;

	/** \brief get the number of inputs in the network
	 *
	 * \return the number of inputs in the network
	 *
	 * \see getInput()
	 */

	virtual int							getNbInputs() const = 0;

	/** \brief get the input tensor specified by the given index
	 *
	 * \param index the index of the input tensor
	 *
	 * \return the input tensor, or null if the index is out of range
	 *
	 * \see getNbInputs()
	 */
	virtual ITensor*					getInput(int index) const = 0;				// adding inputs invalidates indexing here

	/** \brief get the number of outputs in the network
	 *
	 * \return the number of outputs in the network
	 *
	 * \see getOutput()
	 */

	virtual int							getNbOutputs() const = 0;

	/** \brief get the output tensor specified by the given index
	 *
	 * \param index the index of the output tensor
	 *
	 * \return the output tensor, or null if the index is out of range
	 *
	 * \see getNbOutputs()
	 */
	virtual ITensor*					getOutput(int index) const = 0;				// adding outputs invalidates indexing here


	/** \brief destroy this INetworkDefinition object
	 */
	virtual void						destroy() = 0;
protected:
	virtual ~INetworkDefinition() {}
};




/** \class IProfiler
 *
 * \brief application-implemented interface for profiling
 * 
 * When this class is added to an execution context, the profiler will be called once per layer for each invocation of execute().
 * Note that enqueue() does not currently support profiling.
 *
 * the profiler will only be called after execution is complete. It has a small impact on execution time.
 */
class IProfiler
{
public:
	/** \brief layer time reporting callback
	 * 
	 * \param layerName the name of the layer, set when constructing the network definition
	 * \param ms the time in milliseconds to execute the layer
	 */
	virtual void reportLayerTime(const char* layerName, float ms) = 0;

protected:
	virtual ~IProfiler() {}
};


class ICudaEngine;


/** \class IExecutionContext
 *
 * \brief context for executing inference using an engine
 * 
 * Multiple execution contexts may exist for one ICudaEngine instance, allowing the same 
 * engine to be used for the execution of multiple batches simultaneously.
 * 
 */
class IExecutionContext
{
public:
	
	/**
	 * \brief synchronously execute inference on a batch
	 *
	 * this method requires a array of input and output buffers. The mapping from tensor names to indices can be queried using ICudaEngine::getBindingIndex()
	 * \param batchSize the batch size. This is at most the value supplied when the engine was built.
	 * \param bindings an array of pointers to input and output buffers for the network.
	 *
	 * \return true if execution succeeded
	 * \see ICudaEngine::getBindingIndex() ICudaEngine::getMaxBatchSize()
	 */
	virtual bool execute(int batchSize, void** bindings) = 0;

	/**
	* \brief asynchronously execute inference on a batch
	*
	* this method requires a array of input and output buffers. The mapping from tensor names to indices can be queried using ICudaEngine::getBindingIndex()
	* \param batchSize the batch size. This is at most the value supplied when the engine was built.
	* \param bindings an array of pointers to input and output buffers for the network.
	* \param stream a cuda stream on which the inference kernels will be enqueued
	* \param inputConsumed an optional event which will be signaled when the input buffers can be refilled with new data
	*
	* \return true if the kernels were enqueued successfully
	* 
	* \see ICudaEngine::getBindingIndex() ICudaEngine::getMaxBatchSize()
	*/
	virtual bool enqueue(int batchSize, void** bindings, cudaStream_t stream, cudaEvent_t* inputConsumed) = 0;

	/** 
	 * \brief set the debug sync flag
	 * 
	 * if this flag is set to true, the engine will log the successful execution for each kernel during execute(). It has no effect when using enqueue().
	 *
	 * \see getDebugSync()
	 */
	virtual void setDebugSync(bool sync) = 0;

	/**
	 * \brief get the debug sync flag
	 *
	 * \see setDebugSync()
	 */
	virtual bool getDebugSync() const = 0;


	/**
	 * \brief set the profiler
	 *
	 * \see IProfiler getProfiler()
	 */
	virtual void setProfiler(IProfiler*) = 0;

	/**
	 * \brief get the profiler
	 *
	 * \see IProfiler setProfiler()
	 */
	virtual IProfiler* getProfiler() const = 0;

	/**
	 * \brief get the associated engine
	 *
	 * \see ICudaEngine
	 */

	virtual const ICudaEngine& getEngine() const = 0;

	/**
	 * \brief destroy this object
	 */
	virtual void destroy() = 0;

protected:
	virtual ~IExecutionContext() {}
};


/** \class ICudaEngine
 *
 * \brief an engine for executing inference on a built network
 *
 */
class ICudaEngine
{
public:
	/** 
	 * \brief get the number of binding indices
	 *
	 * \see getBindingIndex();
	 */
	virtual int getNbBindings() const = 0;

	/** \brief retrieve the binding index for a named tensor
	 * 
	 * IExecutionContext::enqueue() and IExecutionContext::execute() require an array of buffers. 
	 * 
	 * Engine bindings map from tensor names to indices in this array.
	 * Binding indices are assigned at engine build time, and take values in the range [0 ... n-1] where n is the total number of inputs and outputs.
	 *
	 * \param name the tensor name
	 * \return the binding index for the named tensor, or -1 if the name is not found
	 * 
	 * see getNbBindings() getBindingIndex()
	 */
	virtual int getBindingIndex(const char* name) const = 0;

	/** \brief retrieve the name corresponding to a binding index
	 *
	 * this is the reverse mapping to that provided by getBindingIndex()
	 *
	 * \param bindingIndex the binding index
	 * \return the name corresponding to the index, or nullptr if the index is out of range
	 *
	 * \see getBindingIndex()
	 */
	virtual const char* getBindingName(int bindingIndex) const = 0;

	
	/** \brief determine whether a binding is an input binding
	 *
	 * \param bindingIndex the binding index
	 * \return true if the index corresponds to an input binding and the index is in range
	 *
	 * \see getBindingIndex()
	 */
	virtual bool bindingIsInput(int bindingIndex) const = 0;

	/** \brief get the dimensions of a binding
	 *
	 * \param bindingIndex the binding index
	 * \return the dimensions of the binding if the index is in range, otherwise (0,0,0)
	 *
	 * \see getBindingIndex()
	 */
	virtual Dims getBindingDimensions(int bindingIndex) const = 0;

	
	/** \brief determine the required data type for a buffer from its binding index
	 *
	 * \param bindingIndex the binding index
	 * \return the type of the data in the buffer
	 *
	 * \see getBindingIndex()
	 */
	virtual DataType getBindingDataType(int bindingIndex) const = 0;

	/** \brief get the maximum batch size which can be used for inference
	 *
	 * \return the maximum batch size for this engine
	 *
	 * \see getBindingIndex()
	 */
	virtual int getMaxBatchSize() const = 0;

	/** \brief get the number of layers in the network
	 *
	 * the number of layers in the network is not necessarily the number in the original network definition, as layers may be combined or eliminated as the engine is
	 * optimized. This value can be useful when building per-layer tables, such as when aggregating profiling data over a number of executions.
	 *
	 * \return the number of layers in the network
	 * 
	 */
	virtual int getNbLayers() const = 0;

	/** \brief get the amount of workspace the engine uses.
	 *
	 * the workspace size will be no greater than the value provided to the builder when the engine was built, and will typically be smaller.
	 * Workspace will be allocated for each execution context.
	 *
	 */
	virtual std::size_t getWorkspaceSize() const = 0;

	/** \brief serialize the network to a stream
	 *
	 * \return A IHostMemory object that contains the serialized engine.
     *
	 * the network may be deserialized with IRuntime::deserializeCudaEngine()
     *
	 * \see IRuntime::deserializeCudaEngine()
	 */
	virtual IHostMemory* serialize() const = 0;

	/** \brief create an execution context
	 *
	 * \see IExecutionContext.
	 *
	 */
	virtual IExecutionContext* createExecutionContext() = 0;

	/** \brief destroy this object
	 */
	virtual void destroy() = 0;
protected:
	virtual ~ICudaEngine() {}
};

/** enum CalibrationAlgoType
 * 
 * \brief version of calibration algorithm to use 
 */
enum class CalibrationAlgoType : int
{
	kLEGACY_CALIBRATION = 0,
	kENTROPY_CALIBRATION = 1
};
template<> inline int EnumMax<CalibrationAlgoType>() { return 2; }		//!< maximum number of elements in CalibrationAlgoType enum. \see DataType

/** \class IInt8Calibrator
 *
 * \brief application-implemented interface for calibration
 *
 * Calibration is a step performed by the builder when deciding suitable scale factors for 8-bit inference. 
 * 
 * It must also provide a method for retrieving representative images which the calibration process can use to examine 
 * the distribution of activations. It may optionally implement a method for caching the calibration result for reuse
 * on subsequent runs.
 */

class IInt8Calibrator
{
public:
	/** \brief get the batch size used for calibration batches
	 *
	 * \return the batch size
	 */
	virtual int					getBatchSize() const = 0;

	/** \brief get a batch of input for calibration. 
	 * 
	 * The batch size of the input must match the batch size returned by getBatchSize(). 
	 *
	 * \param bindings an array of pointers to device memory that must be set to the memory containing each network input data 
	 * \param names the names of the network input for each pointer in the binding array
	 * \param nbBindings the number of pointers in the bindings array
	 * \return false if there are no more batches for calibration.
	 *
	 * 
	 * \see getBatchSize()
	 */
	virtual bool				getBatch(void* bindings[], const char* names[], int nbBindings) = 0;		// get a pointer to the input batch
	

	/** \brief load a calibration cache.
	 *
	 * calibration is potentially expensive, so it can be useful to generate the calibration data once, then use it on subsequent builds
	 * of the network. The cache includes the regression cutoff and quantile values used to generate it, and will not be used if
	 * these do not batch the settings of the current calibrator. However, the network should also be recalibrated if its structure
	 * changes, or the input data set changes, and it is the responsibility of the application to ensure this.
	 *
	 * \param length the length of the cached data, that should be set by the called function. If there is no data, this should be zero.	
	 *
	 * \return a pointer to the cache, or nullptr if there is no data
	 */
	virtual const void*			readCalibrationCache(std::size_t& length) = 0;

	/** \brief save a calibration cache
	 * 
	 * \param ptr a pointer to the data to cache
	 * \param length the length in bytes of the data to cache
	 *
	 * \see readCalibrationCache.
	 */
	virtual void				writeCalibrationCache(const void* ptr, std::size_t length) = 0;

	/** \brief get the algorithm used by this calibrator
	*
	* \return the algorithm used by the calibrator
	*/
	virtual CalibrationAlgoType getAlgorithm() = 0;
protected:
	virtual ~IInt8Calibrator() {}
};


/** Entropy calibrator. This is the preferred calibrator, as it is less complicated than the legacy calibrator and produces better results
*/

class IInt8EntropyCalibrator : public IInt8Calibrator
{
	/** Signal that this is the entropy calibrator	*/
	virtual CalibrationAlgoType getAlgorithm() { return CalibrationAlgoType::kENTROPY_CALIBRATION; }
protected:
	virtual ~IInt8EntropyCalibrator() {}
};


/** legacy calibrator for compatibility with 2.0 EA. Will be removed in 2.2
 * \deprecated
 */

class IInt8LegacyCalibrator : public IInt8Calibrator
{
public:
	/** Signal that this is the legacy calibrator	*/
	virtual CalibrationAlgoType getAlgorithm() { return CalibrationAlgoType::kENTROPY_CALIBRATION; }

	/** \brief the quantile (between 0 and 1) that will be used to select the region maximum when the quantile method is in use
	*
	* see the user guide for more details on how the quantile is used.
	*/
	virtual double				getQuantile() const = 0;

	/** \brief the fraction (between 0 and 1) of the maximum used to define the regression cutoff when using regression to determine the region maximum
	*
	* see the user guide for more details on how the regression cutoff is used
	*/
	virtual double				getRegressionCutoff() const = 0;

	/** \brief load a histogram
	*
	* histogram generation is potentially expensive, so it can be useful to generate the histograms once, then use them when exploring
	* the space of calibrations. The histograms should be regenerated if the network structure
	* changes, or the input data set changes, and it is the responsibility of the application to ensure this.
	*
	* \param length the length of the cached data, that should be set by the called function. If there is no data, this should be zero.
	*
	* \return a pointer to the cache, or nullptr if there is no data
	*/
	virtual const void*			readHistogramCache(std::size_t& length) = 0;

	/** \brief save a histogram cache.
	*
	* \param ptr a pointer to the data to cache
	* \param length the length in bytes of the data to cache
	*
	* \see readHistogramCache
	*/
	virtual void				writeHistogramCache(const void* ptr, std::size_t length) = 0;
protected:
	virtual ~IInt8LegacyCalibrator() {}
};

/**
* \class IBuilder
*
* \brief builds an engine from a network definition
*
*/
class IBuilder
{
public:
	/** \brief create a network definition object.
	 * 
	 * \see INetworkDefinition
	 */

	virtual nvinfer1::INetworkDefinition* createNetwork() = 0;

	/** \brief set the maximum batch size
	 * 
	 * \param batchSize the maximum batch size which can be used at execution time, and also the batch size for which the engine will be optimized
	 *
	 * \see getMaxBatchSize()
	 */
	virtual void setMaxBatchSize(int batchSize) = 0;

	/** \brief get the maximum batch size
 	 * 
	 * \return the maximum batch size
	 *
	 * \see setMaxBatchSize()
	 */
	virtual int getMaxBatchSize() const = 0;


	/** \brief set the maximum workspace size
	 *
	 * \param workspaceSize the maximum GPU temporary memory which the engine can use at execution time
	 *
	 * \see getMaxWorkspaceSize()
	 */
	virtual void setMaxWorkspaceSize(std::size_t workspaceSize) = 0;

	/** \brief get the maximum workspace size
	 *
	 * \return the maximum workspace size
	 *
	 * \see setMaxWorkspaceSize()
	 */
	virtual std::size_t getMaxWorkspaceSize() const = 0;


	/** \brief set whether half2 mode is used
	 *
	 * half2 mode is a paired-image mode that is significantly faster for batch sizes greater than one on platforms with fp16 support
	 * 
	 * \param mode whether half2 mode is used
	 *
	 * \see getHalf2Mode()
	 */
	virtual void setHalf2Mode(bool mode) = 0;

	/** \brief query whether half2 mode is used
	 * \see setHalf2Mode()
	 */
	virtual bool getHalf2Mode() const = 0;

	/** \brief set whether the builder should use debug synchronization
	 *
	 * if this flag is true, the builder will synchronize after timing each layer, and report the layer name. It can be useful when diagnosing issues at build time.
	 */
	virtual void setDebugSync(bool sync) = 0;

	/** \brief query whether the builder will use debug synchronization
	 *
	 * \see setDebugSync()
	 */
	virtual bool getDebugSync() const = 0;


	/** \brief set the number of minimization iterations used when timing layers
	 *
	 * When timing layers, the builder minimizes over a set of average times for layer execution. This parameter controls the number of iterations
	 * used in minimzation.
	 *
	 * \see getMinFindIterations()
	 */
	virtual void setMinFindIterations(int minFind) = 0;

	/** query the number of minimzation iterations
	 * 
	 * \see setMinFindIterations()
	 */
	virtual int getMinFindIterations() const = 0;

	/** \brief set the number of minimization iterations used when timing layers
	 *
	 * When timing layers, the builder minimizes over a set of average times for layer execution. This parameter controls the number of iterations
	 * used in averaging.
	 *
	 * \see getAverageFindIterations()
	 */
	virtual void setAverageFindIterations(int avgFind) = 0;

	/** query the number of averaging iterations
	 *
	 * \see setMinFindIterations()
	 */
	virtual int getAverageFindIterations() const = 0;

	/** \brief build a CUDA engine from a network definition
	 *
	 * \see INetworkDefinition ICudaEngine
	 */

	virtual nvinfer1::ICudaEngine* buildCudaEngine(nvinfer1::INetworkDefinition& network) = 0;

	/** \brief determine whether the platform has fast native fp16
	 */

	virtual bool platformHasFastFp16() const = 0;

	/** \brief determine whether the platform has fast native int8
	 */

	virtual bool platformHasFastInt8() const = 0;

	/** \brief destroy this object
	 */

	virtual void destroy() = 0;

	/** \brief set the maximum value for a region
	*
	* used for INT8 mode compression
	*/

	virtual void setInt8Mode(bool mode) = 0;

	/** \brief query whether Int8 mode is used
	* \see setInt8Mode()
	*/
	virtual bool getInt8Mode() const = 0;

	/** \brief set Int8 Calibration interface
	*/

	virtual void setInt8Calibrator(IInt8Calibrator* calibrator) = 0;

protected:
	virtual ~IBuilder() {}
};


/**
* \class IPluginFactory
*
* \brief plugin factory for deserialization
*
*/
class IPluginFactory
{
public:
	/**
	 * \brief create a plugin from serialized data
	 *
	 * \param layerName the name of the layer
	 * \param serialData the serialized data
	 * \param serialLength the length of the serialized data
	 *
	 * \return the plugin
	 *
	 * \see IPlugin::serialize()
	 */
	virtual IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) = 0;
};


/**
* \class IRuntime
*
* \brief allows a serialized engine to be deserialized
*
*/
class IRuntime
{
public:
	/** \brief deserialize an engine from a stream
	 * 
	 * \param blob the memory that holds the serialized engine.
     * \param size the size of the memory
	 * \param pluginFactory the plugin factory, if any plugins are used by the network, otherwise nullptr 
	 *
	 * \return the engine, or nullptr if it could not be deserialized
	 */

	virtual nvinfer1::ICudaEngine* deserializeCudaEngine(const void *blob, std::size_t size, IPluginFactory* pluginFactory) = 0;

	/** \brief destroy this object
	 */
	virtual void destroy() = 0;
protected:
	~IRuntime() {}
};


/**
 * \class ILogger
 *
 * \brief application-implemented logging interface for the builder, engine and runtime. 
 * 
 * Note that although a logger is passed on creation to each instance of a IBuilder or IRuntime interface, the logger is internally considered a singleton, and thus
 * multiple instances of IRuntime and/or IBuilder must all use the same logger
 * 
 */

class ILogger
{
public:
	/**
	* \enum Severity
	*
	* The severity corresponding to a log message
	*
	*/
	enum class Severity
	{
		kINTERNAL_ERROR = 0,		//!< An internal error has occurred. Execution is unrecoverable
		kERROR = 1,					//!< An application error has occurred
		kWARNING = 2,				//!< An application error has been discovered, but TensorRT has recovered or fallen back to a default
		kINFO = 3					//!< Informational messages
	};

	/**
	 * a callback implemented by the application to handle logging messages
	 *
	 * \param severity the severity of the message
	 * \param msg the log message, null terminated.
	 */
	virtual void log(Severity severity, const char* msg) = 0;
protected:
	virtual ~ILogger() {}
};
template<> inline int EnumMax<ILogger::Severity>() { return 4; }		//!< maximum number of elements in DataType enum. \see DataType

} // namespace nvinfer1


extern "C" void* createInferBuilder_INTERNAL(void* logger, int version);	//!< internal C entry point for creating IBuilder
extern "C" void* createInferRuntime_INTERNAL(void* logger, int version);	//!< internal C entry point for creating IRuntime

/**
* \brief return the logger object 
*/

extern "C" nvinfer1::ILogger* getLogger();

/**
* \brief return the library version number
*
* The format is as for TENSORRT_VERSION: (TENSORRT_MAJOR * 1000) + (TENSORRT_MINOR * 100) + TENSOR_PATCH
*
*/

extern "C" int getInferLibVersion();

namespace nvinfer1
{
/**
* \brief create an instance of an IBuilder class
*
* This class is the logging class for the builder, engine and runtime
*
*/
namespace // unnamed namespace in case the compiler doesn't inline these
{
inline IBuilder* createInferBuilder(ILogger& logger)
{
	return static_cast<IBuilder*>(createInferBuilder_INTERNAL(&logger, NV_TENSORRT_VERSION));
}



/**
* \brief create an instance of an IRuntime class
*
* This class is the logging class for the builder, engine and runtime
*
*/
inline IRuntime* createInferRuntime(ILogger& logger)
{
	return static_cast<IRuntime*>(createInferRuntime_INTERNAL(&logger, NV_TENSORRT_VERSION));
}


}


}
#endif
