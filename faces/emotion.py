import sys
from matplotlib import pyplot as plt
import numpy as np
import keras
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import cv2

EMOTIONS = ["angry", "scared", "happy", "sad", "surprised","neutral"]

model = load_model('models/emotion/epoch_90.hdf5')
face_image = cv2.imread(sys.argv[1])
face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
face_image = cv2.resize(face_image, (48,48))
face_image = face_image.astype('float') / 255.0
cv2.imshow('test', face_image)
cv2.waitKey(0)
face_image = img_to_array(face_image)
face_image = np.expand_dims(face_image, axis=0)
preds = model.predict(face_image)[0]
label = EMOTIONS[preds.argmax()]
print(label)

canvas = np.zeros((220, 300, 3), dtype="uint8")
for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
    # construct the label text
    text = "{}: {:.2f}%".format(emotion, prob * 100)
    # draw the label + probability bar on the canvas
    w = int(prob * 300)
    cv2.rectangle(canvas, (5, (i * 35) + 5),(w, (i * 35) + 35), (0, 0, 255), -1)
    cv2.putText(canvas, text, (10, (i * 35) + 23),cv2.FONT_HERSHEY_SIMPLEX, 0.45,(255, 255, 255), 2)
cv2.imshow("Probabilities", canvas)
cv2.waitKey(0)
