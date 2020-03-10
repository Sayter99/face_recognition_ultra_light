#!/usr/bin/env python3
import sys
import numpy as np
import keras
import rospkg
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import cv2
import rospy
from face_recognition_ultra_light.msg import EmotionOutput
from sensor_msgs.msg import CompressedImage
import tensorflow as tf
from tensorflow.python.keras.backend import set_session

graph = tf.get_default_graph()
sess = tf.Session()
set_session(sess)

class emotion_recognition():

    def __init__(self):
        rospy.init_node('emotion_recognition', anonymous=True)
        self.EMOTIONS = ["angry", "scared", "happy", "sad", "surprised","neutral"]
        self.emotion_pub = rospy.Publisher('/emotion/output', EmotionOutput, queue_size=10)
        pkg_path = rospkg.RosPack().get_path('face_recognition_ultra_light')
        self.model = load_model(pkg_path + '/faces/models/emotion/epoch_90.hdf5')
        rospy.Subscriber('/emotion/face', CompressedImage, self.callback, queue_size=5)

    def callback(self, data):
        global graph
        global sess
        with graph.as_default():
            set_session(sess)
            encoded_data = np.fromstring(data.data, np.uint8)
            face_image = cv2.imdecode(encoded_data, cv2.IMREAD_UNCHANGED)
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            face_image = cv2.resize(face_image, (48,48))
            face_image = face_image.astype('float') / 255.0
            face_image = img_to_array(face_image)
            face_image = np.expand_dims(face_image, axis=0)
            preds = self.model.predict(face_image)[0]
            label = self.EMOTIONS[preds.argmax()]
            msg = EmotionOutput()
            msg.name = data.header.frame_id
            msg.emotion = label
            self.emotion_pub.publish(msg)
        
    def start(self):
        rospy.spin()

if __name__ == '__main__':
    node = emotion_recognition()
    node.start()