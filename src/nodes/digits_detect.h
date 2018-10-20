/**
 * @file	digits_detect.h
 * @author	Carroll Vance
 * @brief	nVidia DIGITS Detection ROS Driver
 *
 * Copyright (c) 2018 Carroll Vance.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef DIGITS_DETECT_H_
#define DIGITS_DETECT_H_

#include <chrono>

#include "cv_bridge/cv_bridge.h"
#include "jetson_tensorrt/ClassifiedRegionOfInterest.h"
#include "jetson_tensorrt/ClassifiedRegionsOfInterest.h"
#include "ros/package.h"
#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/image_encodings.h"

#include "CUDAPipeline.h"
#include "DIGITSDetector.h"

namespace jetson_tensorrt {

class ROSDIGITSDetector {
public:
  ROSDIGITSDetector(ros::NodeHandle nh, ros::NodeHandle nh_private);
  void imageCallback(const sensor_msgs::Image::ConstPtr &msg);

private:
  /* TensorRT */
  jetson_tensorrt::CUDAPipeline *preprocessPipeline = nullptr;
  jetson_tensorrt::LocatedExecutionMemory tensor_input, tensor_output;
  jetson_tensorrt::DIGITSDetector *engine = nullptr;

  /* ROS */
  ros::Publisher region_pub;
  ros::Subscriber image_sub;

  std::vector<std::string> classes;

  /* Params */
  float threshold;
  std::string model_path, cache_path, weights_path, classes_path;
  std::string image_subscribe_topic;
  nvinfer1::DataType data_type;
  int model_image_depth, model_image_width, model_image_height,
      model_num_classes;
  double mean_1, mean_2, mean_3;

  std::chrono::time_point<std::chrono::system_clock> start_t;
  int frames;

  ros::NodeHandle nh;
  ros::NodeHandle nh_private;
};

} // namespace jetson_tensorrt
#endif /* DIGITS_DETECT_H_ */
