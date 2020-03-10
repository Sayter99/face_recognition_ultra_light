#!/usr/bin/env python3
# @Author: fyr91
# @Date: 2019-10-22 15:05:15
# @Last Modified by: sayter
# @Last Modified time: 2020-03-06 22:04:25
import os
import cv2
import dlib
import rospy
import rospkg
import tensorflow as tf
import numpy as np
import pickle
import onnx
import onnxruntime as ort
from onnx_tf.backend import prepare
from imutils import face_utils
from sensor_msgs.msg import CompressedImage
from astra_camera.srv import SetUVCExposure
from cv_bridge import CvBridge, CvBridgeError

def area_of(left_top, right_bottom):
    """
    Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]

def iou_of(boxes0, boxes1, eps=1e-5):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)

def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Perform hard non-maximum-supression to filter out boxes with iou greater
    than threshold
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
        picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]

def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.5, top_k=-1):
    """
    Select boxes that contain human faces
    Args:
        width: original image width
        height: original image height
        confidences (N, 2): confidence array
        boxes (N, 4): boxes array in corner-form
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
        boxes (k, 4): an array of boxes kept
        labels (k): an array of labels for each boxes kept
        probs (k): an array of probabilities for each boxes being in corresponding labels
    """
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(box_probs,
        iou_threshold=iou_threshold,
        top_k=top_k,
        )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

class face_detection():
    def __init__(self, topic_name):
        rospy.init_node('face_detection', anonymous=True)
        pkg_path = rospkg.RosPack().get_path('face_recognition_ultra_light')
        self.onnx_path = pkg_path + '/onnx/ultra_light_640.onnx'
        self.onnx_model = onnx.load(self.onnx_path)
        # onnx.checker.check_model(self.onnx_model)
        # onnx.helper.printable_graph(self.onnx_model.graph)
        self.predictor = prepare(self.onnx_model)
        self.ort_session = ort.InferenceSession(self.onnx_path)
        self.input_name = self.ort_session.get_inputs()[0].name

        self.shape_predictor = dlib.shape_predictor(pkg_path + '/faces/models/facial_landmarks/shape_predictor_5_face_landmarks.dat')
        self.fa = face_utils.facealigner.FaceAligner(self.shape_predictor, desiredFaceWidth=112, desiredLeftEye=(0.3, 0.3))
        self.threshold = 0.63
        # load distance
        with open(pkg_path + "/faces/embeddings/embeddings.pkl", "rb") as f:
            (self.saved_embeds, self.names) = pickle.load(f)

        tf.reset_default_graph()
        self.sess = tf.Session()
        self.saver = tf.train.import_meta_graph(pkg_path + '/faces/models/mfn/m1/mfn.ckpt.meta')
        self.saver.restore(self.sess, pkg_path + '/faces/models/mfn/m1/mfn.ckpt')

        self.images_placeholder = tf.get_default_graph().get_tensor_by_name('input:0')
        self.embeddings = tf.get_default_graph().get_tensor_by_name('embeddings:0')
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name('phase_train:0')
        self.embedding_size = self.embeddings.get_shape()[1]
        
        rospy.wait_for_service('/camera/set_uvc_exposure')
        try:
            set_exposure = rospy.ServiceProxy('/camera/set_uvc_exposure', SetUVCExposure)
            response = set_exposure(300)
            print(response)
        except rospy.ServiceException:
            print('Service call failed')
        self.bridge = CvBridge()
        self.face_pub = rospy.Publisher('/emotion/face', CompressedImage, queue_size=10)
        rospy.Subscriber(topic_name, CompressedImage, self.compressedCallback, queue_size=3)
        rospy.on_shutdown(self.shutdownCb)

    def compressedCallback(self, image):
        encoded_data = np.fromstring(image.data, np.uint8)
        decoded_image = cv2.imdecode(encoded_data, cv2.IMREAD_UNCHANGED)
        h, w, _ = decoded_image.shape
        preprocessed_image = cv2.cvtColor(decoded_image, cv2.COLOR_BGR2RGB)
        mean = np.array([127, 127, 127])
        preprocessed_image = (preprocessed_image - mean) / 128
        preprocessed_image = np.transpose(preprocessed_image, [2, 0, 1])
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
        preprocessed_image = preprocessed_image.astype(np.float32)
        confidences, boxes = self.ort_session.run(None, {self.input_name: preprocessed_image})
        boxes, labels, probs = predict(w, h, confidences, boxes, 0.7)
        faces = []
        boxes[boxes<0] = 0
        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            x1, y1, x2, y2 = box
            gray = cv2.cvtColor(decoded_image, cv2.COLOR_RGB2GRAY)
            aligned_face = self.fa.align(decoded_image, gray, dlib.rectangle(left = x1, top=y1, right=x2, bottom=y2))
            aligned_face = cv2.resize(aligned_face, (112,112))

            aligned_face = aligned_face - 127.5
            aligned_face = aligned_face * 0.0078125

            faces.append(aligned_face)
        if len(faces) > 0:
            predictions = []
            faces = np.array(faces)
            feed_dict = {self.images_placeholder: faces, self.phase_train_placeholder:False}
            embeds = self.sess.run(self.embeddings, feed_dict=feed_dict)
            # prediciton using distance
            for embedding in embeds:
                diff = np.subtract(self.saved_embeds, embedding)
                dist = np.sum(np.square(diff), 1)
                idx = np.argmin(dist)
                if dist[idx] < self.threshold:
                    predictions.append(self.names[idx])
                else:
                    predictions.append("unknown")

            # draw
            for i in range(boxes.shape[0]):
                box = boxes[i, :]

                text = predictions[i]

                x1, y1, x2, y2 = box
                crop_image = decoded_image[y1:y2, x1:x2]

                msg = CompressedImage()
                msg = self.bridge.cv2_to_compressed_imgmsg(crop_image)
                msg.header.frame_id = text
                self.face_pub.publish(msg)

                # Draw a label with a name below the face
                cv2.rectangle(decoded_image, (x1, y1), (x2, y2), (80,18,236), 2)
                cv2.rectangle(decoded_image, (x1, y2 - 20), (x2, y2), (80,18,236), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(decoded_image, text, (x1 + 6, y2 - 6), font, 0.3, (255, 255, 255), 1)
        cv2.imshow('result', decoded_image)
        cv2.waitKey(1)
    
    def shutdownCb(self):
        # self.sess.close()
        pass

    def start(self):
        rospy.spin()

if __name__ == '__main__':
    face_detection = face_detection('/camera/rgb/image_rect_color/compressed')
    face_detection.start()
