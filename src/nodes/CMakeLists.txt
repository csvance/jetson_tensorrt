add_executable(
    digits_classify
    digits_classify.cpp
)

add_executable(
    digits_detect
    digits_detect.cpp
)

add_executable(
    tensorflow_generic
    tensorflow_generic.cpp
)

target_link_libraries(digits_classify jetson_tensorrt)
target_link_libraries(digits_detect jetson_tensorrt)
target_link_libraries(tensorflow_generic jetson_tensorrt)
