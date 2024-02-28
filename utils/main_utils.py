from tqdm import tqdm
import cv2
import numpy as np
import time
from utils.yolo_utils import detect_humans
from utils.image_utils import resize_frame, apply_background_subtraction, process_accumulated_heatmap, display_fps_text


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
