import yolov9
import cv2
import numpy as np

def initialize_yolov9(weights_path, iou_conf, iou_thresh, class_id):
    # Check if GPU is available
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        model = yolov9.load(weights_path, device="cuda")

        # Set model parameters
        model.conf = iou_conf  # NMS confidence threshold
        model.iou = iou_thresh  # NMS IoU threshold
        model.classes = [class_id]  # Class id for person

        return model

    else:
        print("GPU is not available. Using CPU for YOLOv9 inference.")
        model = yolov9.load(weights_path, device="cpu")

        # Set model parameters
        model.conf = iou_conf  # NMS confidence threshold
        model.iou = iou_thresh  # NMS IoU threshold
        model.classes = [class_id]  # Class id for person

        return model

def detect_humans_yolov9(frame, model, confidence_threshold=0.5, display_boxes=False):
    # perform inference on frame
    results = model(frame)

    # parse results
    predictions = results.pred[0]
    boxes = predictions[:, :4]  # x1, y1, x2, y2s
    confidence = predictions[:, 4]  # confidence scores

    # filter predictions based on confidence threshold
    filtered_boxes = [np.array(box[:4], dtype=int) for box, conf in zip(boxes, confidence) if conf > confidence_threshold]

    if display_boxes:
        # draw bounding boxes on frame
        for box in filtered_boxes:
            x1, y1, x2, y2 = np.array(box[:4], dtype=int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)

    return filtered_boxes
