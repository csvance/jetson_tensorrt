# ros_jetson_tensorrt
This repository is in heavy active development, but will eventually contain ROS nodes specifically designed to utilize TensorRT on Jetson. It currently houses an abstraction layer which hides the host/device memory management paradigm required by TensorRT which will be implemented into ROS nodes.

# Planned Network Support
- [DIGITS][digits] - ImageNet, DetectNet, SegNet
- Caffe - Generic
- Tensorflow - Generic

# Documentation
- [Doxygen][docs]

# Test Graphs
- [Download][test_graphs]
- Caffe - GoogLeNet
- Tensorflow - Inception v3

[digits]: https://github.com/NVIDIA/DIGITS
[docs]: https://csvance.github.io/ros_jetson_tensorrt/
[test_graphs]: https://www.dropbox.com/s/t4mso4qwa64dsh7/models.zip?dl=0
