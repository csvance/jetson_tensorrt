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

 #include "DigitsDetector.h"
 #include "ros/ros.h"

 DIGITSDetector engine;

 void imageCallback(const sensor_msgs::Image::ConstPtr& msg)
{

}


 int main(int argc, char **argv)
 {

   ros::init(argc, argv, "digits_detect");
   ros::NodeHandle n;

   std::string model_path = n.getParam("model_path");
   std::string weights_path = n.getParam("weights_path");

   std::string cache_path;
   n.param("cache_path", cache_path, "detection.tensorcache");

   int image_depth, image_width, image_height;
   n.param("image_depth", &image_depth, 1);
   n.param("image_width", &image_width, 1024);
   n.param("image_height", &image_height, 512);

   ROS_INFO("Loading nVidia DIGITS model...");
   engine = DIGITSDetector(model_path, weights_path, cache_path,
       image_depth, image_width, image_height, num_classes);

   std::string image_subscribe_topic;
   n.param("image_subscribe_topic", image_subscribe_topic, "camera");

   ros::Subscriber image_sub = n.subscribe<sensor_msgs::Image>(image_subscribe_topic, 5, imageCallback);
   ros::Publisher region_pub = n.advertise<jetson_tensorrt_msgs::CategorizedRegionsOfInterest>("detections", 5);

   ros::spin();

   return 0;
 }
