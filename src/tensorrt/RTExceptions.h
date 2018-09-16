/**
 * @file	RTExceptions.h
 * @author	Carroll Vance
 * @brief	Contains exceptions used by the TensorRT layer
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

#ifndef EXCEPTIONS_H_
#define EXCEPTIONS_H_

#include <string>
#include <exception>
#include <stdexcept>
#include <sstream>

namespace jetson_tensorrt{

template <typename T>
std::string to_string(T value)
{
	//create an output string stream
	std::ostringstream os ;

	//throw the value into the string stream
	os << value ;

	//convert the string stream into a string and return
	return os.str() ;
}

/**
 * @brief	Exception thrown when a new TensorRT model fails to build
 */
class ModelBuildException: public std::invalid_argument {
private:
    std::string message_;
public:
    ~ModelBuildException() throw() {}
    explicit ModelBuildException(const std::string& message);
    virtual const char* what() const throw() {
        return message_.c_str();
    }
};

/**
 * @brief	Exception thrown when an existing TensorRT model fails to deserialize
 */
class ModelDeserializeException: public std::invalid_argument {
private:
    std::string message_;
public:
   ~ModelDeserializeException() throw() {}
    explicit ModelDeserializeException(const std::string& message);
    virtual const char* what() const throw() {
        return message_.c_str();
    }
};

/**
 * @brief	Exception thrown when an existing TensorRT model fails to serialize
 */
class ModelSerializeException: public std::invalid_argument {
private:
    std::string message_;
public:
    ~ModelSerializeException() throw() {}
    explicit ModelSerializeException(const std::string& message);
    virtual const char* what() const throw() {
        return message_.c_str();
    }
};

/**
 * @brief	Exception thrown a host memory allocation call fails
 */
class HostMemoryAllocException: public std::bad_alloc {
private:
    std::string message_;
public:
    ~HostMemoryAllocException() throw() {}
    explicit HostMemoryAllocException(const std::string& message);
    virtual const char* what() const throw() {
        return message_.c_str();
    }
};

/**
 * @brief	Exception thrown a CUDA device memory allocation call fails
 */
class DeviceMemoryAllocException: public std::bad_alloc {
private:
    std::string message_;
public:
    ~DeviceMemoryAllocException() throw() {}
    explicit DeviceMemoryAllocException(const std::string& message);
    virtual const char* what() const throw() {
        return message_.c_str();
    }
};


/**
 * @brief	Exception thrown a CUDA device memory free call fails
 */
class DeviceMemoryFreeException: public std::invalid_argument {
private:
    std::string message_;
public:
    ~DeviceMemoryFreeException() throw() {}
    explicit DeviceMemoryFreeException(const std::string& message);
    virtual const char* what() const throw() {
        return message_.c_str();
    }
};

/**
 * @brief	Exception thrown memory fails to copy from the host to a CUDA device
 */
class HostDeviceTransferException: public std::exception {
private:
    std::string message_;
public:
    ~HostDeviceTransferException() throw() {}
    explicit HostDeviceTransferException(const std::string& message);
    virtual const char* what() const throw() {
        return message_.c_str();
    }
};

/**
 * @brief	Exception thrown memory fails to copy from a CUDA device to the host
 */
class DeviceHostTransferException: public std::exception {
private:
    std::string message_;
public:
    ~DeviceHostTransferException() throw() {}
    explicit DeviceHostTransferException(const std::string& message);
    virtual const char* what() const throw() {
        return message_.c_str();
    }
};

/**
 * @brief	Exception thrown when input/output dimensions registered do not match the loaded model
 */
class ModelDimensionMismatchException: public std::invalid_argument {
private:
    std::string message_;
public:
    ~ModelDimensionMismatchException() throw() {}
    explicit ModelDimensionMismatchException(const std::string& message);
    virtual const char* what() const throw() {
        return message_.c_str();
    }
};

/**
 * @brief	Exception thrown when batch is larger than models maximum batch size
 */
class BatchSizeException: public std::invalid_argument {
private:
    std::string message_;
public:
    ~BatchSizeException() throw() {}
    explicit BatchSizeException(const std::string& message);
    virtual const char* what() const throw() {
        return message_.c_str();
    }
};

/**
 * @brief	Exception thrown if the TensorRT engine fails to execute during a prediction
 */
class EngineExecutionException: public std::exception {
private:
    std::string message_;
public:
    ~EngineExecutionException() throw() {}
    explicit EngineExecutionException(const std::string& message);
    virtual const char* what() const throw() {
        return message_.c_str();
    }
};

/**
 * @brief	Exception thrown if an unsupported parameter is passed to an RTEngine
 */
class UnsupportedConfigurationException: public std::invalid_argument {
private:
    std::string message_;
public:
    ~UnsupportedConfigurationException() throw() {}
    explicit UnsupportedConfigurationException(const std::string& message);
    virtual const char* what() const throw() {
        return message_.c_str();
    }
};

/**
 * @brief	Exception thrown if an error occurs during image preprocessing
 */
class PreprocessorException: public std::exception {
private:
    std::string message_;
public:
    ~PreprocessorException() throw() {}
    explicit PreprocessorException(const std::string& message);
    virtual const char* what() const throw() {
        return message_.c_str();
    }
};


}

#endif /* EXCEPTIONS_H_ */
