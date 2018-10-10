/**
 * @file	digits_detect.cpp
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

#include "digits_detect.h"

namespace jetson_tensorrt {

void ROSDIGITSDetector::imageCallback(const sensor_msgs::Image::ConstPtr &msg) {

  /* 0. Initialize */
  if (engine == nullptr) {

    ROS_INFO("Loading nVidia DIGITS model...");
    engine = new DIGITSDetector(model_path, weights_path, cache_path,
                                model_image_depth, model_image_width,
                                model_image_height, model_num_classes);
    ROS_INFO("Done loading nVidia DIGITS model.");

    tensor_input = engine->allocInputs(MemoryLocation::DEVICE, true);
    tensor_output = engine->allocOutputs(MemoryLocation::UNIFIED);

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

    // Statistics
    start_t = std::chrono::system_clock::now();
    frames = 0;
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

  cv_bridge::CvImagePtr cv_ptr;
  if (debug)
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);

  float x_scale = (float)msg->width / (float)model_image_width;
  float y_scale = (float)msg->height / (float)model_image_height;

  for (std::vector<ClassRectangle>::iterator it = rects.begin();
       it != rects.end(); ++it) {
    CategorizedRegionOfInterest region;
    region.x = (*it).x;
    region.y = (*it).y;
    region.w = (*it).w;
    region.h = (*it).h;
    region.id = (*it).id;

    if (debug) {
      ROS_INFO("DETECT(%d): %d,%d %dx%d", region.id, (int)(region.x * x_scale),
               (int)(region.y * y_scale), (int)(region.w * x_scale),
               (int)(region.h * y_scale));

      cv::Rect rect((int)(region.x * x_scale), (int)(region.y * y_scale),
                    (int)(region.w * x_scale), (int)(region.h * y_scale));
      cv::rectangle(cv_ptr->image, rect, cv::Scalar(0, 255, 0), 8);
    }

    regions.regions.push_back(region);
  }
  region_pub.publish(regions);

  if (debug)
    debug_pub.publish(cv_ptr->toImageMsg());

  frames++;
  if (frames > 0 && frames % 10 == 0) {
    ROS_INFO("%f FPS",
             (1000 * (float)frames) /
                 std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::system_clock::now() - start_t)
                     .count());

    start_t = std::chrono::system_clock::now();
    frames = 0;
  }
}

ROSDIGITSDetector::ROSDIGITSDetector(ros::NodeHandle nh,
                                     ros::NodeHandle nh_private) {

  std::string package_path = ros::package::getPath("jetson_tensorrt");

  nh_private.param("model_path", model_path,
                   package_path + std::string("/networks/detectnet.prototxt"));
  nh_private.param("weights_path", weights_path,
                   package_path + std::string("/networks/ped-100.caffemodel"));
  nh_private.param("cache_path", cache_path,
                   package_path +
                       std::string("/networks/detection.tensorcache"));

  ROS_INFO("model_path: %s", model_path.c_str());
  ROS_INFO("weights_path: %s", weights_path.c_str());
  ROS_INFO("cache_path: %s", cache_path.c_str());

  nh_private.param("model_image_depth", model_image_depth,
                   (int)DIGITSDetector::DEFAULT::DEPTH);
  nh_private.param("model_image_width", model_image_width,
                   (int)DIGITSDetector::DEFAULT::WIDTH);
  nh_private.param("model_image_height", model_image_height,
                   (int)DIGITSDetector::DEFAULT::HEIGHT);
  nh_private.param("model_num_classes", model_num_classes,
                   (int)DIGITSDetector::DEFAULT::CLASSES);

  nh_private.param("mean1", mean_0, (float)0.0);
  nh_private.param("mean2", mean_1, (float)0.0);
  nh_private.param("mean3", mean_2, (float)0.0);

  nh_private.param("threshold", threshold, (float)0.5);

  nh_private.param("debug", debug, 0);

  nh_private.param("image_subscribe_topic", image_subscribe_topic,
                   std::string("/csi_cam/image_raw"));

  ROS_INFO("image_subscribe_topic: %s", image_subscribe_topic.c_str());

  image_sub = nh.subscribe<sensor_msgs::Image>(
      image_subscribe_topic, 2, &ROSDIGITSDetector::imageCallback, this);

  if (debug)
    debug_pub = nh_private.advertise<sensor_msgs::Image>("debug_output", 5);

  region_pub =
      nh_private.advertise<jetson_tensorrt::CategorizedRegionsOfInterest>(
          "detections", 5);

  this->nh = nh;
  this->nh_private = nh_private;
}

} // namespace jetson_tensorrt
