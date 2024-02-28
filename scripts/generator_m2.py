import os
import cv2
import time
import numpy as np
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
    # Check if GPU is available
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        net = cv2.dnn.readNet(WEIGHTS_PATH, CFG_PATH)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        print("Using GPU for YOLO inference.")
    else:
        net = cv2.dnn.readNet(WEIGHTS_PATH, CFG_PATH)
        print("Using CPU for YOLO inference.")

    classes = []
    with open(COCO_NAMES, "r") as f:
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
            if confidence > confidence_threshold and class_id == 0: # 0 id for person
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

    return [boxes[i] for i in indices.reshape(-1)] if indices is not None else []

def resize_frame(frame, resize_factor=0.5):
    return cv2.resize(frame, None, fx=resize_factor, fy=resize_factor)

def apply_background_subtraction(fgbg, roi_frame):
    gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(gray_roi)
    
    thresh = 2
    maxValue = 2
    _, thresholded = cv2.threshold(fgmask, thresh, maxValue, cv2.THRESH_BINARY)
    return thresholded

def process_accumulated_heatmap(frame, accum_image):
    color_image = cv2.applyColorMap(accum_image, cv2.COLORMAP_HOT)
    return cv2.addWeighted(frame, 0.7, color_image, 0.7, 0)

def display_fps_text(frame, frame_count, start_time):
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
    return frame

def process_frames(cap, fgbg, yolo_net, yolo_classes, yolo_layer_names, display_fps=False, resize_factor=0.5, skip_frames=2, yolo_skip_frames=3):
    first_iteration_indicator = True
    first_frame = None
    accum_image = None
    frame_count = 0
    yolo_frame_count = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm(total=total_frames, desc='Processing Frames', unit='frames')

    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        progress_bar.update(1)

        if not ret:
            print("End of video.")
            break

        frame_count += 1
        yolo_frame_count += 1

        # Optimization 3: Skip Frames
        if frame_count % skip_frames != 0:
            continue

        frame = resize_frame(frame, resize_factor)

        if first_iteration_indicator:
            first_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape[:2]
            accum_image = np.zeros((height, width), dtype=np.uint8)
            first_iteration_indicator = False
        else:
            # Optimization 4: Optimized YOLO Detection
            if yolo_frame_count % yolo_skip_frames == 0:
                human_boxes = detect_humans(frame, yolo_net, yolo_layer_names, display_boxes=True)
                yolo_frame_count = 0
            else:
                human_boxes = []  # Use the previous boxes

            for box in human_boxes:
                x, y, w, h = box
                roi_frame = frame[y:y + h, x:x + w]
                
                # Background subtraction and motion accumulation within the bounding box
                thresholded = apply_background_subtraction(fgbg, roi_frame)
                accum_image[y:y + h, x:x + w] = cv2.add(accum_image[y:y + h, x:x + w], thresholded)

            # Display live heatmap
            live_heatmap = process_accumulated_heatmap(frame, accum_image)

            if display_fps:
                live_heatmap = display_fps_text(live_heatmap, frame_count, start_time)

            cv2.imshow('Live Heatmap', live_heatmap)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    progress_bar.close()
    return first_frame, accum_image

def generate_overlay(first_frame, accum_image):
    color_image = cv2.applyColorMap(accum_image, cv2.COLORMAP_HOT)
    result_overlay = cv2.addWeighted(first_frame, 0.7, color_image, 0.7, 0)
    return result_overlay

def save_image(output_folder, result_overlay, empty_output_dir=False):
    output_path = os.path.join(output_folder, 'heatmap_overlay.jpg')

    if empty_output_dir:
        # Empty the existing content inside the output directory
        for file_name in os.listdir(output_folder):
            file_path = os.path.join(output_folder, file_name)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error: {e}")

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

    first_frame, accum_image = process_frames(cap, fgbg, yolo_net, yolo_classes, yolo_layer_names, display_fps=True)
    result_overlay = generate_overlay(first_frame, accum_image)

    output_folder = './output'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_path = save_image(output_folder, result_overlay, empty_output_dir=True)

    cv2.imshow('Final Heatmap Overlay Output', result_overlay)
    print(f"Final overlay image saved at: {output_path}")

    cap.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

