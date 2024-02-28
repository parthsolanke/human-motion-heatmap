import os
import cv2
from util import yolo_utils, image_utils, main_utils

COCO_NAMES = r"./yolo-coco/coco.names"
CFG_PATH = r"./yolo-coco/yolov3/yolov3.cfg"
WEIGHTS_PATH = r"./yolo-coco/yolov3/yolov3.weights"
VIDEO_PATH = r"./data/input.mp4"

def main():
    video_path = VIDEO_PATH
    cap = image_utils.initialize_video_capture(video_path)
    fgbg = cv2.createBackgroundSubtractorMOG2()

    # Initialize YOLO for human detection
    yolo_net, yolo_classes, yolo_layer_names = yolo_utils.initialize_yolo(COCO_NAMES, CFG_PATH, WEIGHTS_PATH)

    first_frame, accum_image = main_utils.process_frames(cap, fgbg, yolo_net, yolo_classes, yolo_layer_names, display_fps=True, skip_frames=2, yolo_skip_frames=5)
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
