/**
 * @file	digits_classify.cpp
 * @author	Carroll Vance
 * @brief	nVidia DIGITS Classification ROS Node
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

#include "digits_classify.h"
#include <fstream>
#include <streambuf>
#include <string>

namespace jetson_tensorrt {

void ROSDIGITSClassifier::imageCallback(
    const sensor_msgs::Image::ConstPtr &msg) {

  /* 0. Initialize */
  if (engine == nullptr) {

    ROS_INFO("Loading nVidia DIGITS model...");
    engine = new DIGITSClassifier(model_path, weights_path, cache_path,
                                  model_image_depth, model_image_width,
                                  model_image_height, model_num_classes);
    ROS_INFO("Done loading nVidia DIGITS model.");

    tensor_input = engine->allocInputs(MemoryLocation::DEVICE, true);
    tensor_output = engine->allocOutputs(MemoryLocation::UNIFIED);

    if (msg->encoding.compare(sensor_msgs::image_encodings::RGB8) == 0) {
      preprocessPipeline = CUDAPipeline::createRGBImageNetPipeline(
          msg->width, msg->height, model_image_width, model_image_height,
          make_float3(mean_1, mean_2, mean_3));
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
  std::vector<RTClassification> classifications =
      engine->classify(tensor_input, tensor_output, threshold);

  /* 3. Publish */
  Classifications msg_classifications;

  cv_bridge::CvImagePtr cv_ptr;
  if (debug)
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);

  for (std::vector<RTClassification>::iterator it = classifications.begin();
       it != classifications.end(); ++it) {

    Classification classification;
    classification.id = it->id;
    classification.confidence = it->confidence;

    if (it->id < classes.size())
      classification.desc = classes[it->id];
    else
      classification.desc = "";

    if (debug) {

      std::string debug_msg =
          classes[it->id] + " : " + std::to_string(classification.confidence);

      ROS_INFO("CLASSIFY(%d): %s", classification.id, debug_msg.c_str());

      putText(cv_ptr->image, debug_msg.c_str(), cvPoint(30, 30),
              CV_FONT_HERSHEY_SIMPLEX, 0.8, cvScalar(200, 200, 250), 1, CV_AA,
              true);
    }

    msg_classifications.classifications.push_back(classification);
  }

  classification_pub.publish(msg_classifications);

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

ROSDIGITSClassifier::ROSDIGITSClassifier(ros::NodeHandle nh,
                                         ros::NodeHandle nh_private) {

  std::string package_path = ros::package::getPath("jetson_tensorrt");

  nh_private.param("debug", debug, 0);

  nh_private.param("model_path", model_path,
                   package_path + std::string("/networks/googlenet.prototxt"));
  nh_private.param("weights_path", weights_path,
                   package_path +
                       std::string("/networks/bvlc_googlenet.caffemodel"));
  nh_private.param("cache_path", cache_path,
                   package_path +
                       std::string("/networks/classify.tensorcache"));
  nh_private.param("sysnet_words_path", sysnet_words_path,
                   package_path +
                       std::string("/networks/ilsvrc12_synset_words.txt"));

  std::ifstream words_file(sysnet_words_path.c_str());
  std::string text((std::istreambuf_iterator<char>(words_file)),
                   std::istreambuf_iterator<char>());

  std::string delim = "\n";
  auto start = 0U;
  auto end = text.find(delim);
  int index = 0;
  while (end != std::string::npos) {

    if (debug)
      ROS_INFO("%d: %s", index, text.substr(start, end - start).c_str());

    classes.push_back(text.substr(start, end - start));

    start = end + delim.length();
    end = text.find(delim, start);
    index++;
  }

  ROS_INFO("model_path: %s", model_path.c_str());
  ROS_INFO("weights_path: %s", weights_path.c_str());
  ROS_INFO("cache_path: %s", cache_path.c_str());
  ROS_INFO("sysnet_words_path: %s", sysnet_words_path.c_str());

  nh_private.param("model_image_depth", model_image_depth,
                   (int)DIGITSClassifier::DEFAULT::DEPTH);
  nh_private.param("model_image_width", model_image_width,
                   (int)DIGITSClassifier::DEFAULT::WIDTH);
  nh_private.param("model_image_height", model_image_height,
                   (int)DIGITSClassifier::DEFAULT::HEIGHT);
  nh_private.param("model_num_classes", model_num_classes,
                   (int)DIGITSClassifier::DEFAULT::CLASSES);

  nh_private.param("mean1", mean_1, 0.0);
  nh_private.param("mean2", mean_2, 0.0);
  nh_private.param("mean3", mean_3, 0.0);

  nh_private.param("threshold", threshold, (float)0.5);

  nh_private.param("image_subscribe_topic", image_subscribe_topic,
                   std::string("/csi_cam/image_raw"));

  ROS_INFO("image_subscribe_topic: %s", image_subscribe_topic.c_str());

  image_sub = nh.subscribe<sensor_msgs::Image>(
      image_subscribe_topic, 2, &ROSDIGITSClassifier::imageCallback, this);

  if (debug)
    debug_pub = nh_private.advertise<sensor_msgs::Image>("debug_output", 5);

  classification_pub = nh_private.advertise<jetson_tensorrt::Classifications>(
      "classifications", 100);

  this->nh = nh;
  this->nh_private = nh_private;
}

} // namespace jetson_tensorrt
