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
import json 
from std_msgs.msg import String
from tensorflow.python.keras.backend import set_session

graph = tf.compat.v1.get_default_graph()
sess = tf.compat.v1.Session()
set_session(sess)

class emotion_recognition():

    def __init__(self):
        rospy.init_node('emotion_recognition', anonymous=True)
        self.EMOTIONS = ["angry", "scared", "happy", "sad", "surprised","neutral"]
        self.emotion_pub = rospy.Publisher('/emotion/output', EmotionOutput, queue_size=10)
        self.emotion_pub_json = rospy.Publisher('/emotion/output_json', String, queue_size=10)
        self.emotion_pub_total_json = rospy.Publisher('/emotion/output_total_json', String, queue_size=10)


        pkg_path = rospkg.RosPack().get_path('face_recognition_ultra_light')
        self.model = load_model(pkg_path + '/faces/models/emotion/epoch_90.hdf5')
        rospy.Subscriber('/emotion/face', CompressedImage, self.callback, queue_size=5)
        self.queue = []
        self.num_samples = 20
        self.emotion_totals = [0, 0, 0, 0, 0, 0]


    def callback(self, data):
        global graph
        global sess
        with graph.as_default():
            set_session(sess)
            encoded_data = np.fromstring(data.data, np.uint8)
            self.process_face_image(encoded_data)
            emotion_idx = self.publish_emotions(data)
            self.publish_emotions_json()
            self.publish_emotions_total_json(emotion_idx)


    def publish_emotions_total_json(self, emotion_idx):
        self.emotion_totals[emotion_idx] = self.emotion_totals[emotion_idx] + 1
        vals = {
            self.EMOTIONS[0]: str(self.emotion_totals[0]),
            self.EMOTIONS[1]: str(self.emotion_totals[1]),
            self.EMOTIONS[2]: str(self.emotion_totals[2]),
            self.EMOTIONS[3]: str(self.emotion_totals[3]),
            self.EMOTIONS[4]: str(self.emotion_totals[4]),
            self.EMOTIONS[5]: str(self.emotion_totals[5])
        }
        vals_json = json.dumps(vals) 
        self.emotion_pub_total_json.publish(vals_json)

    def process_face_image(self, encoded_data): 
        face_image = cv2.imdecode(encoded_data, cv2.IMREAD_UNCHANGED)
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        face_image = cv2.resize(face_image, (48,48))
        face_image = face_image.astype('float') / 255.0
        face_image = img_to_array(face_image)
        face_image = np.expand_dims(face_image, axis=0)
        self.preds = self.model.predict(face_image)[0]


    def publish_emotions(self,data):
        # Filter to get the most likely emotion from the mode in past values
        # I keep a running total of the last num_samples emotions we have detected
        self.queue.append(self.preds.argmax())
        if len(self.queue) > self.num_samples:
            self.queue.pop(0)
        max_idx =  max(set(self.queue), key=self.queue.count)
        label = self.EMOTIONS[max_idx]

        msg = EmotionOutput()
        msg.name = data.header.frame_id
        msg.emotion = label
        self.emotion_pub.publish(msg)
        return max_idx


    def publish_emotions_json(self):
        # Package data as json string and publish
        vals = {
            self.EMOTIONS[0]: str(self.preds[0]),
            self.EMOTIONS[1]: str(self.preds[1]),
            self.EMOTIONS[2]: str(self.preds[2]),
            self.EMOTIONS[3]: str(self.preds[3]),
            self.EMOTIONS[4]: str(self.preds[4]),
            self.EMOTIONS[5]: str(self.preds[5])
        }
        vals_json = json.dumps(vals) 
        self.emotion_pub_json.publish(vals_json)

        
    def start(self):
        rospy.spin()

if __name__ == '__main__':
    node = emotion_recognition()
    node.start()