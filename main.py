import os
import cv2
from util import image_utils, main_utils, yolov9_utils

WEIGHTS_PATH = r"./yolo-coco/yolov9/gelan-c.pt"
VIDEO_PATH = r"./data/input.mp4"

def main():
    video_path = VIDEO_PATH
    cap = image_utils.initialize_video_capture(video_path)
    fgbg = cv2.createBackgroundSubtractorMOG2()

    # Initialize YOLO V9 for human detection
    model = yolov9_utils.initialize_yolov9(WEIGHTS_PATH, iou_conf=0.25, iou_thresh=0.45, class_id=0)

    first_frame, accum_image = main_utils.process_frames_yolov9(cap, fgbg, yolo_model=model, display_fps=True, skip_frames=1, yolo_skip_frames=5)
    result_overlay = main_utils.generate_overlay(first_frame, accum_image)

    output_folder = './output'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_utils.save_image(output_folder, result_overlay, empty_output_dir=True)

    cv2.imshow('Final Heatmap Overlay Output', result_overlay)
    print(f"Final overlay image saved at: {os.path.join(output_folder, 'heatmap_overlay.jpg')}")

    cap.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
