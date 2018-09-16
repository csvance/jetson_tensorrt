add_executable(
  test_caffe
  caffe.cpp
)
target_link_libraries(test_caffe jetson_tensorrt)

add_executable(
  test_digits_classification
  classification.cpp
)
target_link_libraries(test_digits_classification jetson_tensorrt)

add_executable(
  test_digits_detection
  detection.cpp
)
target_link_libraries(test_digits_detection jetson_tensorrt)

add_executable(
  test_tensorflow
  tensorflow.cpp
)
target_link_libraries(test_tensorflow jetson_tensorrt)

add_custom_command(
        TARGET test_caffe POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_BINARY_DIR}/test_caffe
                /home/nvidia/test_caffe
                )
