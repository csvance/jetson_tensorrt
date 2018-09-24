/**
 * @file	digits_detect.cpp
 * @author	Carroll Vance
 * @brief	nVidia DIGITS Detection ROS Node
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

#include "digits_detect.h"
#include "jetson_tensorrt/CategorizedRegionOfInterest.h"
#include "jetson_tensorrt/CategorizedRegionsOfInterest.h"
#include "ros/ros.h"
#include "sensor_msgs/Image.h"

#include "CUDAPipeline.h"
#include "DIGITSDetector.h"

using namespace jetson_tensorrt;

/* TensorRT */
CUDAPipeline *preprocessPipeline = nullptr;
LocatedExecutionMemory input, output;
DIGITSDetector engine;

/* ROS */
ros::Publisher region_pub;

/* Params */
float threshold;
std::string model_path, cache_path, weights_path;
std::string image_subscribe_topic;
int model_image_depth, model_image_width, model_image_height, model_num_classes;

void imageCallback(const sensor_msgs::Image::ConstPtr &msg) {
  /* 1. Preprocess */
  if (preprocessPipeline == nullptr) {
    if (msg.encoding == sensor_msgs::image_encodings::RGB8) {
      preprocessPipeline = createRGBImageNetPipeline(
          msg.width, msg.height, model_image_width, model_image_height);
    } else if (msg.encoding == sensor_msgs::image_encodings::YUV422) {
      // TODO: Implement YUV422 preprocess pipeline
      ROS_INFO("Unsupported image encoding: %s", msg.encoding);
    }
  } else {
    ROS_INFO("Unsupported image encoding: %s", msg.encoding);
    return;
  }

  CUDAPipeIO input =
      CUDAPipeIO(MemoryLocation::HOST, msg.data, msg.height * msg.step);

  CUDAPipeIO output = preprocessPipeline->pipe(input);

  input.batch[0][0] = output.data;

  /* 2. Inference */
  std::vector<ClassRectangle> rects = engine.predict(input, output, threshold);

  /* 3. Publish */
  CategorizedRegionsOfInterest regions;

  for (std::vector<T>::iterator it = rects.begin(); it != rects.end(); ++it) {
    CategorizedRegionOfInterest region;
    region.x = (*it).x;
    region.y = (*it).y;
    region.w = (*it).w;
    region.h = (*it).h;
    region.id = (*it).id;

    regions.regions.add(region);
  }

  region_pub.publish(regions);
}

int main(int argc, char **argv) {

  ros::init(argc, argv, "digits_detect");
  ros::NodeHandle nh;

  nh.getParam("model_path", model_path);
  nh.getParam("weights_path", weights_path);
  nh.getParam("cache_path", cache_path);

  nh.param("model_image_depth", model_image_depth,
           (int)DIGITSDetector::DEFAULT::DEPTH);
  nh.param("model_image_width", model_image_width,
           (int)DIGITSDetector::DEFAULT::WIDTH);
  nh.param("model_image_height", model_image_height,
           (int)DIGITSDetector::DEFAULT::HEIGHT);
  nh.param("model_num_classes", model_num_classes,
           (int)DIGITSDetector::DEFAULT::CLASSES);

  nh.param("threshold", threshold, 0.5);

  ROS_INFO("Loading nVidia DIGITS model...");
  engine =
      DIGITSDetector(model_path, weights_path, cache_path, model_image_depth,
                     model_image_width, model_image_height, model_num_classes);

  input = engine.allocInputs(MemoryLocation::HOST);
  output = engine.allocOutputs(MemoryLocation::HOST);

  nh.param("image_subscribe_topic", image_subscribe_topic,
           std::string("camera"));

  ros::Subscriber image_sub =
      nh.subscribe<sensor_msgs::Image>(image_subscribe_topic, 5, imageCallback);
  region_pub = nh.advertise<jetson_tensorrt::CategorizedRegionsOfInterest>(
      "detections", 5);

  ros::spin();

  return 0;
}
