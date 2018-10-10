add_executable(
    digits_classify
    digits_classify.cpp
)

add_executable(
    digits_detect
    digits_detect.cpp
    digits_detect_node.cpp
)

add_executable(
    tensorflow_generic
    tensorflow_generic.cpp
)

add_library(
    DIGITSDetect
    digits_detect.cpp
    digits_detect_nodelet.cpp
  )

target_link_libraries(DIGITSDetect jetson_tensorrt ${catkin_LIBRARIES})
target_link_libraries(digits_classify jetson_tensorrt ${catkin_LIBRARIES})
target_link_libraries(digits_detect jetson_tensorrt ${catkin_LIBRARIES})
target_link_libraries(tensorflow_generic jetson_tensorrt ${catkin_LIBRARIES})
