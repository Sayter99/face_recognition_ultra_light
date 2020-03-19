#!/usr/bin/env python
import roslib
import rospy
import tf
import sys
import numpy as np
import rospkg
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage, Image, CameraInfo

class face_distance():
    def __init__(self):
        rospy.init_node('face_distance', anonymous=True)
        depth_info = rospy.wait_for_message('/camera/depth_registered/camera_info', CameraInfo)
        rgb_info = rospy.wait_for_message('/camera/rgb/camera_info', CameraInfo)
        rospy.Subscriber('/emotion/face', CompressedImage, self.compressedCb, queue_size=1)
        rospy.Subscriber('/camera/depth/image_raw', Image, self.depthCb, queue_size=1)
        self.bridge = CvBridge()
        self.depth = None
        self.br = tf.TransformBroadcaster()
        self.uvc_fx = rgb_info.P[0]
        self.uvc_fy = rgb_info.P[5]
        self.uvc_cx = rgb_info.P[2]
        self.uvc_cy = rgb_info.P[6]
        self.depth_fx = depth_info.P[0]
        self.depth_fy = depth_info.P[5]
        self.depth_cx = depth_info.P[2]
        self.depth_cy = depth_info.P[6]

    def compressedCb(self, data):
        name = data.header.frame_id.split()
        x = float(name[1])
        y = float(name[2])
        if self.depth is not None and name[0] != 'unknown':
            diff_x = (x - self.uvc_cx)
            diff_y = (y - self.uvc_cy)
            # index_y = max(0, min(int(x), 400 - 1))
            # index_x = max(0, min(int(y) + 40, 640 - 1))
            index_y = max(5, min(int(self.depth_cy + diff_y), 400 - 6))
            index_x = max(5, min(int(self.depth_cx + diff_x), 640 - 6))
            window = self.depth[index_y - 5 : index_y + 5, index_x - 5 : index_x + 5]
            dist = np.median(window) * 0.001
            real_x = -diff_x * dist / self.depth_fx
            real_y = -diff_y * dist / self.depth_fy
            self.br.sendTransform((dist, real_x, real_y),
                                  tf.transformations.quaternion_from_euler(0, 0, 0),
                                  rospy.Time.now(),
                                  name[0],
                                  'camera_link')

    def depthCb(self, img):
        self.depth = self.bridge.imgmsg_to_cv2(img, '16UC1')

    def start(self):
        rospy.spin()

if __name__ == '__main__':
    face_distance = face_distance()
    face_distance.start()
