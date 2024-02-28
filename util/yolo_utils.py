import cv2
import numpy as np


def initialize_yolo(coco_names, cfg_path, weights_path):
    # Check if GPU is available
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        net = cv2.dnn.readNet(weights_path, cfg_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        print("Using GPU for YOLO inference.")
    else:
        net = cv2.dnn.readNet(weights_path, cfg_path)
        print("Using CPU for YOLO inference.")

    classes = []
    with open(coco_names, "r") as f:
        classes = [line.strip() for line in f]
    layer_names = net.getUnconnectedOutLayersNames()

    return net, classes, layer_names


def detect_humans(frame, net, layer_names, confidence_threshold=0.5, display_boxes=False):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)

    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold and class_id == 0:  # 0 id for person
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

                if display_boxes:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1, cv2.LINE_AA)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)

    return [boxes[i] for i in indices.flatten()] if indices is not None else []
