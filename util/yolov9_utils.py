import torch
import cv2
import numpy as np
import yolov9
from yolov9.models.common import DetectMultiBackend
from yolov9.utils.general import non_max_suppression, scale_boxes
from yolov9.utils.torch_utils import select_device, smart_inference_mode
from yolov9.utils.augmentations import letterbox
import PIL.Image

@smart_inference_mode()
def detect_humans_yolov9(image_path, weights, imgsz=640, conf_thres=0.1, iou_thres=0.45, device='0', data='data/coco.yaml'):
    # Initialize
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, fp16=False, data=data)
    stride, names, pt = model.stride, model.names, model.pt

    # Load image
    image = PIL.Image.open(image_path)
    img0 = np.array(image)
    assert img0 is not None, f'Image Not Found {image_path}'
    img = letterbox(img0, imgsz, stride=stride, auto=True)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=False, visualize=False)

    # Apply NMS
    pred = non_max_suppression(pred[0][0], conf_thres, iou_thres, classes=None, max_det=1000)

    # Process detections
    detected_boxes = []
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                detected_boxes.append(xyxy)

    return detected_boxes

# You can add additional functions or modify as needed in yolov9_utils.py
