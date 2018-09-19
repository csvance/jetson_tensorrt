add_subdirectory(cuda)

add_library(
    jetson_tensorrt
    CaffeRTEngine.cpp
    CUDANode.cpp
    CUDANodeIO.cpp
    CUDAPipeline.cpp
    ClusteredNonMaximumSuppression.cpp
    DIGITSClassifier.cpp
    DIGITSDetector.cpp
    NetworkIO.cpp
    RTCommon.cpp
    TensorRTEngine.cpp
    TensorflowRTEngine.cpp
)

target_link_libraries(jetson_tensorrt jetson_tensorrt_cuda -lnvinfer -lnvparsers -lnvinfer_plugin)
