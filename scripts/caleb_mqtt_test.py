#!/usr/bin/env python3
import paho.mqtt.client as mqtt
import time
# Done using the tutorial: https://www.youtube.com/watch?v=QAaXNt0oqSI

# functions called by the client are done async and will not print in the order they are called. 

# when ?anything is done?
def on_log(client, userdata, level, buf):
    print("log: ", buf)

# when connection is done
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("connected OK")
    else:
        print("Bad connection return code = ", rc)

def on_disconnect(client, userdata, flags, rc=0):
    print("disconnected return code = ", str(rc))

def on_message(client, userdata, msg):
    topic=msg.topic
    m_decode=str(msg.payload.decode("utf-8",))
    print("Message: ", m_decode, " Topic:" , topic)


broker = "broker.hivemq.com" # where we want to send our message to

client = mqtt.Client("face_detection") # name must be unique?

# created links to the functions created at the top of the file
client.on_connect = on_connect
#client.on_log = on_log # Normally run without the log file, but if more informatoin is needed can uncomment
client.on_disconnect = on_disconnect
client.on_message = on_message

print("Connecting to broker ", broker) 
client.connect(broker)


# need to loop in ordre for the logs to be processed
client.loop_start()

# publish a message inside of the loop
# takes in arguments topic name, message
client.subscribe("emotion")
client.publish("emotion", "happy")

time.sleep(4)
client.loop_stop()

client.disconnect()