#!/usr/bin/env python3
import paho.mqtt.client as mqtt
import json

def on_subscribe(client, userdata, mid, granted_qos):
    print("Subscribed: " + str(mid) + " " + str(granted_qos))

# process a message when recived
def on_message(client, userdata, msg):
    m_decode = str(msg.payload.decode("utf-8"))
    emotion_dict = json.loads(m_decode)
    for key in emotion_dict:
        print(key, '->', emotion_dict[key])
    print('')


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("connected OK")
    else:
        print("Bad connection return code = ", rc)


client_name = "batbot"
broker = "broker.hivemq.com"
client = mqtt.Client(client_name)


# Add all methods defined at the start of the file to the client
client.on_subscribe = on_subscribe
client.on_message = on_message
client.on_connect = on_connect

client.connect(broker)
client.subscribe("face/emotion", qos=1)

client.loop_forever()
