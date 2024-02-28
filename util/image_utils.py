import os
import cv2
import time


def initialize_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()
    return cap


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
