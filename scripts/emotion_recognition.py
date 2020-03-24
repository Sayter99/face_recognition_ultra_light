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
import json 
from std_msgs.msg import String

graph = tf.compat.v1.get_default_graph()
sess = tf.compat.v1.Session()
set_session(sess)

# make a queue to hold the last x emotions to smooth out how people are feeing


class emotion_recognition():

    def __init__(self):
        rospy.init_node('emotion_recognition', anonymous=True)
        self.EMOTIONS = ["angry", "scared", "happy", "sad", "surprised","neutral"]
        self.emotion_pub = rospy.Publisher('/emotion/output', EmotionOutput, queue_size=10)
        self.emotion_pub_json = rospy.Publisher('/emotion/output_json', String, queue_size=10)

        pkg_path = rospkg.RosPack().get_path('face_recognition_ultra_light')
        self.model = load_model(pkg_path + '/faces/models/emotion/epoch_90.hdf5')
        rospy.Subscriber('/emotion/face', CompressedImage, self.callback, queue_size=5)
        self.queue = []
        self.num_samples = 20

    def callback(self, data):
        global graph
        global sess
        global queue
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
            #label = self.EMOTIONS[preds.argmax()]

            # Filter to get the most likely emotion from the mode in past values
            self.queue.append(preds.argmax())
            if len(self.queue) > self.num_samples:
                self.queue.pop(0)
            max_idx =  max(set(self.queue), key=self.queue.count)
            label = self.EMOTIONS[max_idx]

            msg = EmotionOutput()
            msg.name = data.header.frame_id
            msg.emotion = label
            self.emotion_pub.publish(msg)

            # Package data as json string and publish
            # TODO: I wanna loop though the emotions and do this automatically 
            vals = {
                self.EMOTIONS[0]: str(preds[0]),
                self.EMOTIONS[1]: str(preds[1]),
                self.EMOTIONS[2]: str(preds[2]),
                self.EMOTIONS[3]: str(preds[3]),
                self.EMOTIONS[4]: str(preds[4]),
                self.EMOTIONS[5]: str(preds[5])
            }
            vals_json = json.dumps(vals)
            # TODO: conform that the json format is correct. 
            self.emotion_pub_json.publish(vals_json)



        
    def start(self):
        rospy.spin()

if __name__ == '__main__':
    node = emotion_recognition()
    node.start()