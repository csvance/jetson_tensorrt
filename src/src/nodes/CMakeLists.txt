add_executable(
    digits_classify
    digits_classify.cpp
)

add_executable(
    digits_detect
    digits_detect.cpp
)

target_link_libraries(digits_classify jetson_tensorrt)
