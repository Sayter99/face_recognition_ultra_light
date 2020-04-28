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
import paho.mqtt.client as mqtt
from datetime import datetime 
import os
import urllib.parse as urlparse


graph = tf.compat.v1.get_default_graph()
sess = tf.compat.v1.Session()
set_session(sess)

def on_connect(client, userdata, flags, rc):
        print("CONNACK received with code %d." % (rc))

class emotion_recognition():
    def __init__(self):
        rospy.init_node('emotion_recognition', anonymous=True)
        self.EMOTIONS = ["angry", "scared", "happy", "sad", "surprised","neutral"]
        self.emotion_pub = rospy.Publisher('/emotion/output', EmotionOutput, queue_size=10)
        pkg_path = rospkg.RosPack().get_path('face_recognition_ultra_light')
        self.model = load_model(pkg_path + '/faces/models/emotion/epoch_90.hdf5')
        rospy.Subscriber('/emotion/face', CompressedImage, self.callback, queue_size=5)

        self.queue = []
        self.num_samples = 5
        self.emotion_totals = [0, 0, 0, 0, 0, 0]
        self.emotion_percentages = [0, 0, 0, 0, 0, 0]
        self.person_name = "unknown"
        self.create_mqtt_client()


    def callback(self, data):
        global graph
        global sess
        
        with graph.as_default():
            # Process face for emotion
            set_session(sess)
            encoded_data = np.fromstring(data.data, np.uint8)
            self.process_face_image(encoded_data)
            emotion_idx = self.publish_emotions(data)

            # Publish emotion data
            self.person_name = data.header.frame_id.split()[0]
            # to publish <name emotion> pair, use publish_emotion_mqtt
            # to publish json data, use publish_emotions_average_mqtt
            self.publish_emotion_mqtt(emotion_idx)


    def create_mqtt_client(self):
        url_str = os.environ.get('CLOUDMQTT_URL')
        url = urlparse.urlparse(url_str)

        self.client = mqtt.Client('batbot_1')
        self.client.on_connect = on_connect
        self.client.username_pw_set(url.username, url.password)
        self.client.connect(url.hostname, url.port)
        self.client.loop_start()


    def publish_emotions_average_mqtt(self, emotion_idx):
        if self.person_name == 'unknown':
            pass
        else:
            self.emotion_totals[emotion_idx] = self.emotion_totals[emotion_idx] + 1
            samples = sum(self.emotion_totals)
            time = datetime.now()
            # This is a Exponential Moving Average that gives more weight to new samples
            for idx, val in enumerate(self.emotion_percentages):
                self.emotion_percentages[idx] -= val/samples
                self.emotion_percentages[idx] += self.preds[idx]/samples

            vals = {
                'time': str(time),
                'name': str(self.person_name),
                self.EMOTIONS[0]: str(self.emotion_percentages[0]),
                self.EMOTIONS[1]: str(self.emotion_percentages[1]),
                self.EMOTIONS[2]: str(self.emotion_percentages[2]),
                self.EMOTIONS[3]: str(self.emotion_percentages[3]),
                self.EMOTIONS[4]: str(self.emotion_percentages[4]),
                self.EMOTIONS[5]: str(self.emotion_percentages[5])
            }
            vals_json = json.dumps(vals) 
            self.client.publish('austin/eye/emotion', vals_json)

    
    def publish_emotion_mqtt(self, emotion_idx):
        if self.person_name == 'unknown':
            pass
        else:
            canvas = np.zeros((220, 300, 3), dtype="uint8")
            for (i, (emotion, prob)) in enumerate(zip(self.EMOTIONS, self.preds)):
                # construct the label text
                text = "{}: {:.2f}%".format(emotion, prob * 100)
                # draw the label + probability bar on the canvas
                w = int(prob * 300)
                cv2.rectangle(canvas, (5, (i * 35) + 5),(w, (i * 35) + 35), (0, 0, 255), -1)
                cv2.putText(canvas, text, (10, (i * 35) + 23),cv2.FONT_HERSHEY_SIMPLEX, 0.45,(255, 255, 255), 2)
            cv2.imshow("Probabilities", canvas)
            cv2.waitKey(1)
            self.queue.append(self.preds.argmax())
            if len(self.queue) > self.num_samples:
                self.queue.pop(0)
            max_idx =  max(set(self.queue), key=self.queue.count)
            label = self.EMOTIONS[max_idx]

            vals = str(self.person_name) + ' ' + str(label)
            print(vals)
            self.client.publish('austin/eye/emotion', vals)


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

        
    def start(self):
        rospy.spin()

if __name__ == '__main__':
    node = emotion_recognition()
    node.start()