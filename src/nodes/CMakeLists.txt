add_executable(
    digits_detect
    digits_detect.cpp
    digits_detect_node.cpp
)
target_link_libraries(digits_detect jetson_tensorrt ${catkin_LIBRARIES})

add_library(
    DIGITSDetect
    digits_detect.cpp
    digits_detect_nodelet.cpp
)
target_link_libraries(DIGITSDetect jetson_tensorrt ${catkin_LIBRARIES})

add_executable(
    digits_classify
    digits_classify.cpp
    digits_classify_node.cpp
)
target_link_libraries(digits_classify jetson_tensorrt ${catkin_LIBRARIES})

add_library(
    DIGITSClassify
    digits_classify.cpp
    digits_classify_nodelet.cpp
)
target_link_libraries(DIGITSClassify jetson_tensorrt ${catkin_LIBRARIES})

add_executable(
    tensorflow_generic
    tensorflow_generic.cpp
)
target_link_libraries(tensorflow_generic jetson_tensorrt ${catkin_LIBRARIES})
