#!/usr/bin/env python

import datetime
from cv_bridge import CvBridge
from threading import Lock

import cv2
import rospy

from jetson_tensorrt.msg import ClassifiedRegionsOfInterest
from jetson_tensorrt.msg import Classifications
from sensor_msgs.msg import Image

class DebugNode(object):
    def __init__(self):
        self._cv_br = CvBridge()

        self._ros_init()

        self._roi_lock = Lock()
        self._latest_rois = []

        self._class_lock = Lock()
        self._latest_classes = []

    def _ros_init(self):
        rospy.init_node('debug')

        rospy.Subscriber('/detector/detections', ClassifiedRegionsOfInterest, self._detect_callback, queue_size=1)
        rospy.Subscriber('/classifier/classifications', Classifications, self._class_callback, queue_size=1)
        rospy.Subscriber(rospy.get_param('image_topic', '/csi_cam/image_raw'), Image, self._camera_callback, queue_size=2)
        self._publisher = rospy.Publisher('rt_debug', Image, queue_size=2)

    def _detect_callback(self, regions):

        self._roi_lock.acquire()
        self._latest_rois = regions.regions
        self._roi_lock.release()

    def _class_callback(self, classes):

        self._class_lock.acquire()
        self._latest_classes = classes.classifications
        self._class_lock.release()

    def _camera_callback(self, image):

        frame = self._cv_br.imgmsg_to_cv2(image)

        self._roi_lock.acquire()
        for roi in self._latest_rois:
            cv2.rectangle(frame, (roi.x, roi.y),
                          (roi.x + roi.w, roi.y + roi.h), (0, 255, 0), 10)
        self._roi_lock.release()

        maxClass = None
        maxConf = 0.0
        self._class_lock.acquire()
        for c in self._latest_classes:
            if c.confidence > maxConf:
                maxConf = c.confidence
                maxClass = c
        self._class_lock.release()

        if maxClass is not None:
            cv2.putText(frame, "%.1f%% : %s" % (maxConf*100, maxClass.desc), (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 6, cv2.LINE_AA)

        self._publisher.publish(self._cv_br.cv2_to_imgmsg(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))


if __name__ == "__main__":
    node = DebugNode()
    rospy.spin()
