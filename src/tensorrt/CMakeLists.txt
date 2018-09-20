add_subdirectory(cuda)

add_library(
    jetson_tensorrt STATIC
    CaffeRTEngine.cpp
    CUDACommon.cpp
    CUDANode.cpp
    CUDAPipeline.cpp
    CUDAPipeNodes.cpp
    DIGITSClassifier.cpp
    DIGITSDetector.cpp
    NetworkDataTypes.cpp
    TensorRTEngine.cpp
    TensorflowRTEngine.cpp
)

target_link_libraries(jetson_tensorrt jetson_tensorrt_cuda -lnvinfer -lnvparsers -lnvinfer_plugin)
