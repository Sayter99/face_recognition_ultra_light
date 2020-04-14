#!/usr/bin/env python3
import os
import urllib.parse as urlparse
import paho.mqtt.client as paho

def on_subscribe(client, userdata, mid, granted_qos):
    print("Subscribed: " + str(mid) + " " + str(granted_qos))

def on_message(client, userdata, msg):
    print(msg.topic + " " + str(msg.qos) + " " + str(msg.payload))


url_str = os.environ.get("CLOUDMQTT_URL")
url = urlparse.urlparse(url_str)

client = paho.Client()
client.on_subscribe = on_subscribe
client.on_message = on_message

# client.connect("broker.mqttdashboard.com", 1883)
client.username_pw_set(url.username, url.password)
client.connect(url.hostname, url.port)

client.subscribe("batbot/emotion", qos=1)


client.loop_forever()
