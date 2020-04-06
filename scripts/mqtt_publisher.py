#!/usr/bin/env python3
import paho.mqtt.client as paho
import time

def on_connect(client, userdata, flags, rc):
    print("CONNACK received with code %d." % (rc))

def on_publish(client, userdata, mid):
    print("mid: " + str(mid))

client = paho.Client()
client.on_connect = on_connect
client.on_publish = on_publish
client.connect("broker.hivemq.com", 1883)
client.loop_start()

while True:
    emotion = "1233211234567"
    (rc, mid) = client.publish("batbot/emotion", str(emotion), qos=1)
    time.sleep(1)
