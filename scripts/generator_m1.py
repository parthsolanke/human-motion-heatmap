import os
import numpy as np
import cv2
from tqdm import tqdm

COCO_NAMES = r"Path to coco.names file"
CFG_PATH = r"Path to cfg file"
WEIGHTS_PATH = r"Path to weights file"
VIDEO_PATH = r"Path to video file"

def initialize_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()
    return cap

def initialize_yolo():
    net = cv2.dnn.readNet(WEIGHTS_PATH, CFG_PATH)
    classes = []
    with open(COCO_NAMES, "r") as f:
        classes = [line.strip() for line in f]
    layer_names = net.getUnconnectedOutLayersNames()
    return net, classes, layer_names

def detect_humans(frame, net, layer_names, confidence_threshold=0.5):
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
            if confidence > confidence_threshold and class_id == 0:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)

    return [boxes[i] for i in indices.flatten()] if indices is not None else []

def process_frames(cap, fgbg, yolo_net, yolo_classes, yolo_layer_names):
    first_iteration_indicator = True
    first_frame = None
    accum_image = None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm(total=total_frames, desc='Processing Frames', unit='frames')

    while True:
        ret, frame = cap.read()
        progress_bar.update(1)

        if not ret:
            print("End of video.")
            break

        if first_iteration_indicator:
            first_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape[:2]
            accum_image = np.zeros((height, width), dtype=np.uint8)
            first_iteration_indicator = False
        else:
            # Human detection using YOLO
            human_boxes = detect_humans(frame, yolo_net, yolo_layer_names)

            for box in human_boxes:
                x, y, w, h = box
                roi_frame = frame[y:y + h, x:x + w]
                
                # Background subtraction and motion accumulation within the bounding box
                gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
                fgmask = fgbg.apply(gray_roi)

                thresh = 2
                maxValue = 2
                _, th1 = cv2.threshold(fgmask, thresh, maxValue, cv2.THRESH_BINARY)
                accum_image[y:y + h, x:x + w] = cv2.add(accum_image[y:y + h, x:x + w], th1)

            # Display live heatmap
            color_image = cv2.applyColorMap(accum_image, cv2.COLORMAP_HOT)
            live_heatmap = cv2.addWeighted(frame, 0.7, color_image, 0.7, 0)
            cv2.imshow('Live Heatmap', live_heatmap)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    progress_bar.close()
    return first_frame, accum_image

def generate_overlay(first_frame, accum_image):
    color_image = cv2.applyColorMap(accum_image, cv2.COLORMAP_HOT)
    result_overlay = cv2.addWeighted(first_frame, 0.7, color_image, 0.7, 0)
    return result_overlay

def save_image(output_folder, result_overlay):
    output_path = os.path.join(output_folder, 'heatmap_overlay.jpg')

    if os.path.isfile(output_path):
        index = 1
        while True:
            new_output_path = os.path.join(output_folder, f'heatmap_overlay_{index}.jpg')
            if not os.path.isfile(new_output_path):
                output_path = new_output_path
                break
            index += 1

    cv2.imwrite(output_path, result_overlay)
    return output_path

def main():
    video_path = VIDEO_PATH
    cap = initialize_video_capture(video_path)
    fgbg = cv2.createBackgroundSubtractorMOG2()

    # Initialize YOLO for human detection
    yolo_net, yolo_classes, yolo_layer_names = initialize_yolo()

    first_frame, accum_image = process_frames(cap, fgbg, yolo_net, yolo_classes, yolo_layer_names)
    result_overlay = generate_overlay(first_frame, accum_image)

    output_folder = 'output'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_path = save_image(output_folder, result_overlay)

    cv2.imshow('Final Heatmap Overlay', result_overlay)
    print(f"Final overlay image saved at: {output_path}")

    cap.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

