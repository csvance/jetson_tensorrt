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
#include "sensor_msgs/image_encodings.h"

#include "CUDAPipeline.h"
#include "DIGITSDetector.h"

using namespace jetson_tensorrt;

/* TensorRT */
CUDAPipeline *preprocessPipeline = nullptr;
LocatedExecutionMemory tensor_input, tensor_output;
DIGITSDetector *engine = nullptr;

/* ROS */
ros::Publisher region_pub;

/* Params */
float threshold;
std::string model_path, cache_path, weights_path;
std::string image_subscribe_topic;
int model_image_depth, model_image_width, model_image_height, model_num_classes;
float mean_0, mean_1, mean_2;

void imageCallback(const sensor_msgs::Image::ConstPtr &msg) {

  /* 0. Initialize */
  if (engine == nullptr) {

    ROS_INFO("Loading nVidia DIGITS model...");
    engine = new DIGITSDetector(model_path, weights_path, cache_path,
                                model_image_depth, model_image_width,
                                model_image_height, model_num_classes);
    ROS_INFO("Done loading nVidia DIGITS model.");

    tensor_input = engine->allocInputs(MemoryLocation::DEVICE, true);
    tensor_output = engine->allocOutputs(MemoryLocation::MAPPED);

    if (msg->encoding.compare(sensor_msgs::image_encodings::RGB8) == 0) {
      preprocessPipeline = CUDAPipeline::createRGBImageNetPipeline(
          msg->width, msg->height, model_image_width, model_image_height,
          make_float3(mean_0, mean_1, mean_2));
    } else if (msg->encoding.compare(sensor_msgs::image_encodings::YUV422) ==
               0) {
      // TODO: Implement YUV422 preprocess pipeline
      ROS_INFO("Unsupported image encoding: %s", msg->encoding.c_str());
      return;
    } else {
      ROS_INFO("Unsupported image encoding: %s", msg->encoding.c_str());
      return;
    }
  }

  /* 1. Preprocess */
  CUDAPipeIO input = CUDAPipeIO(MemoryLocation::HOST, (void *)&msg->data[0],
                                (size_t)(msg->height * msg->step));

  CUDAPipeIO output = preprocessPipeline->pipe(input);

  tensor_input.batch[0][0] = output.data;

  /* 2. Inference */
  std::vector<ClassRectangle> rects =
      engine->detect(tensor_input, tensor_output, threshold);

  /* 3. Publish */
  CategorizedRegionsOfInterest regions;

  for (std::vector<ClassRectangle>::iterator it = rects.begin();
       it != rects.end(); ++it) {
    CategorizedRegionOfInterest region;
    region.x = (*it).x;
    region.y = (*it).y;
    region.w = (*it).w;
    region.h = (*it).h;
    region.id = (*it).id;

    regions.regions.push_back(region);
  }

  region_pub.publish(regions);
}

int main(int argc, char **argv) {

  ros::init(argc, argv, "digits_detect");
  ros::NodeHandle nh("~");

  nh.param("model_path", model_path,
           std::string("/home/nvidia/ros_jetson_tensorrt/detectnet.prototxt"));
  nh.param("weights_path", weights_path,
           std::string("/home/nvidia/ros_jetson_tensorrt/ped-100.caffemodel"));
  nh.param(
      "cache_path", cache_path,
      std::string("/home/nvidia/ros_jetson_tensorrt/detection.tensorcache"));

  ROS_INFO("model_path: %s", model_path.c_str());
  ROS_INFO("weights_path: %s", weights_path.c_str());
  ROS_INFO("cache_path: %s", cache_path.c_str());

  nh.param("model_image_depth", model_image_depth,
           (int)DIGITSDetector::DEFAULT::DEPTH);
  nh.param("model_image_width", model_image_width,
           (int)DIGITSDetector::DEFAULT::WIDTH);
  nh.param("model_image_height", model_image_height,
           (int)DIGITSDetector::DEFAULT::HEIGHT);
  nh.param("model_num_classes", model_num_classes,
           (int)DIGITSDetector::DEFAULT::CLASSES);

  nh.param("mean1", mean_0, (float)0.0);
  nh.param("mean2", mean_1, (float)0.0);
  nh.param("mean3", mean_2, (float)0.0);

  nh.param("threshold", threshold, (float)0.5);

  nh.param("image_subscribe_topic", image_subscribe_topic,
           std::string("/csi_cam/image_raw"));

  ros::Subscriber image_sub =
      nh.subscribe<sensor_msgs::Image>(image_subscribe_topic, 5, imageCallback);
  region_pub = nh.advertise<jetson_tensorrt::CategorizedRegionsOfInterest>(
      "detections", 5);

  ros::spin();

  return 0;
}
