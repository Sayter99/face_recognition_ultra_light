# Face Recognition Ultra Light

The light-weight ROS face recognition package based on Python3. A module of [SMARTBat](https://github.com/ADataDate/SMARTbat).

## Installation

* Install ROS melodic
* [astra_camera](https://github.com/orbbec/ros_astra_camera), we used `stereo_s_u3` by default
* Install python packages, you might need to upgrade your pip version at first
  * `pip3 install -r requirements.txt`

## Emotion Detection over mqtt

* run `roslaunch face_recognition_ultra_light face_recognition_ultra_light.launch`
* `rosrun face_recognition_ultra_light mqtt_emotion_subscriber.py` to receive emotions from the publisher
  * or run `python3 mqtt_emotion_subscriber.py` located in `face_recognition_ultra_light/scripts`
  * we use `mqtt` not `ros` since the special situation under COVID-19 quarantine

## Face Training

* To add training data, you can use `faces/recorder.py`
    ```bash
    cd faces
    python3 recorder.py
    ```
    To quit the program, press **q**
* After having a video for training, you should move it to `faces/training/<name>/<name>.avi`
* Then run `python3 training.py` to create your model
* Finally you can run the launch file
  ```bash
  roslaunch face_recognition_ultra_light face_recognition_ultra_light.launch
  ```
  
## RQt Graph

![](media/rosgraph.png)

## Result

![](media/demo.gif)

## Reference

* https://towardsdatascience.com/real-time-face-recognition-with-cpu-983d35cc3ec5
* https://mc.ai/emotion-recognition-using-keras/
* Chen, Sheng, et al. "Mobilefacenets: Efficient cnns for accurate real-time face verification on mobile devices." Chinese Conference on Biometric Recognition. Springer, Cham, 2018.
* Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).
