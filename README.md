# ros_jetson_tensorrt
Deep learning ROS nodes for nVidias' Jetson platform utilizing TensorRT. **Warning: This repository is in heavy active development**.

## Implemented Nodes
- [DIGITS][digits] - DetectNet

## Planned Nodes
- [DIGITS][digits] - ImageNet, SegNet
- Caffe - Generic
- Tensorflow - Generic
- PyTorch - Generic

## Requirements
- Jetpack 3.3
- CMake

## Build
```
cd ros_jetson_tensorrt
mkdir build && cd build
cmake ..
make
```

## Documentation
- [Doxygen][docs]

## Test Graphs
- [Download][test_graphs]
- Caffe - GoogLeNet
- Tensorflow - Inception v3

[digits]: https://github.com/NVIDIA/DIGITS
[docs]: https://csvance.github.io/ros_jetson_tensorrt/
[test_graphs]: https://www.dropbox.com/s/t4mso4qwa64dsh7/models.zip?dl=0
