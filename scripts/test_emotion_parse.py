#!/usr/bin/env python3
import sys
import rospy
from sensor_msgs.msg import CompressedImage
import json 
from std_msgs.msg import String

class emotion_parser():

    def __init__(self):
        rospy.init_node('emotion_parser', anonymous=True)
        # TODO Question: Why do we set the queue_size to 5? does that have some signifigance?
        rospy.Subscriber('/emotion/output_json', String, self.callback, queue_size=5)

    def callback(self, data):
        emotion_dict = json.loads(data.data)
        for key in emotion_dict:
            print(key, '->', emotion_dict[key])

        
    def start(self):
        rospy.spin()

if __name__ == '__main__':
    node = emotion_parser()
    node.start()