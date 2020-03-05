#!/usr/bin/env python3
import cv2
import rospy
import rospkg
import numpy as np
import onnx
import onnxruntime as ort
from onnx_tf.backend import prepare
from sensor_msgs.msg import CompressedImage

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
    def __init__(self):
        rospy.init_node('face_detection', anonymous=True)
        pkg_path = rospkg.RosPack().get_path('compressed_image_viewer')
        self.onnx_path = pkg_path + '/onnx/ultra_light_640.onnx'
        self.onnx_model = onnx.load(self.onnx_path)
        # onnx.checker.check_model(self.onnx_model)
        # onnx.helper.printable_graph(self.onnx_model.graph)
        self.predictor = prepare(self.onnx_model)
        self.ort_session = ort.InferenceSession(self.onnx_path)
        self.input_name = self.ort_session.get_inputs()[0].name
        rospy.Subscriber('/camera/rgb/image_rect_color/compressed', CompressedImage, self.compressedCallback, queue_size=1)

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
        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            x1, y1, x2, y2 = box
            cv2.rectangle(decoded_image, (x1, y1), (x2, y2), (80,18,236), 2)
            cv2.rectangle(decoded_image, (x1, y2 - 20), (x2, y2), (80,18,236), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            text = f"face: {labels[i]}"
            cv2.putText(decoded_image, text, (x1 + 6, y2 - 6), font, 0.5, (255, 255, 255), 1)
        cv2.imshow('compressed', decoded_image)
        cv2.waitKey(1)
    
    def start(self):
        rospy.spin()

if __name__ == '__main__':
    face_detection = face_detection()
    face_detection.start()
