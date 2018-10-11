# jetson_tensorrt

## Implemented Nodes
### [DIGITS][digits] ImageNet (classification)
- classify_nodes.launch uses the builtin camera on the TX2 and publishes to /classifier/debug_output
#### Parameters

| Param | Type  | Description  |
| :------------- |:-------------| :-----|
| image_subscribe_topic | string | image topic to run classification on |
| model_path | string | absolute path to the model file (.prototxt) |
| weights_path | string | absolute path to the weights file (.caffemodel) |
| cache_path | string | absolute path to the automatically generated tensorcache file |
| sysnet_words_path | string | absolute path to the sysnet words text file |
| model_image_depth | int | model input image depth / number of channels |
| model_image_width | int | model input width in pixels |
| model_image_height | int | model input height in pixels |
| threshold | float | confidence threshold of classifications, between 0.0 and 1.0 |
| mean1, mean2, mean3 | float | ImageNet means |
#### Topics
| Action | Topic | Type |
| :------------- |:-------------| :-----|
| publish | classifications | Classifications |
| subscribe | image_subscribe_topic | Image |

#### Messages
```
# Classification
uint32 id
float32 confidence
string desc
```
```
# Classifications
ClassifiedRegionOfInterest[] regions
```

### [DIGITS][digits] DetectNet (detection)
- detect_nodes.launch uses the builtin camera on the TX2 and publishes to /detector/debug_output
#### Parameters
| Param | Type  | Description  |
| :------------- |:-------------| :-----|
| image_subscribe_topic | string | image topic to run detections on |
| model_path | string | absolute path to the model file (.prototxt) |
| weights_path | string | absolute path to the weights file (.caffemodel) |
| cache_path | string | absolute path to the automatically generated tensorcache file |
| model_image_depth | int | model input image depth / number of channels |
| model_image_width | int | model input width in pixels |
| model_image_height | int | model input height in pixels |
| threshold | float | confidence threshold of detections, between 0.0 and 1.0 |
| mean1, mean2, mean3 | float | ImageNet means |
#### Topics
| Action | Topic | Type |
| :------------- |:-------------| :-----|
| publish | detections | ClassifiedRegionsOfInterest |
| subscribe | image_subscribe_topic | Image |
#### Messages
```
# ClassifiedRegionOfInterest
int32 x
int32 y
int32 w
int32 h
uint32 id
float32 confidence
```
```
# ClassifiedRegionsOfInterest
ClassifiedRegionOfInterest[] regions
```

## Planned Nodes
- [DIGITS][digits] - SegNet
- Caffe - Generic
- Tensorflow - Generic
- PyTorch - Generic

## Requirements
- Jetpack 3.3
- TensorRT 4.0

## Build / Installation
Clone jetson_tensorrt into your catkin_ws/src folder and run catkin_make

## Documentation
- [Doxygen][docs]

## Test Graphs
- [Download][test_graphs]
- Caffe - GoogLeNet
- Tensorflow - Inception v3

[digits]: https://github.com/NVIDIA/DIGITS
[docs]: https://csvance.github.io/jetson_tensorrt/
[test_graphs]: https://www.dropbox.com/s/t4mso4qwa64dsh7/models.zip?dl=0
