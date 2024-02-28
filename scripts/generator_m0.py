import os
import numpy as np
import cv2
from tqdm import tqdm

VIDEO_PATH = r"Path to video file"

def initialize_video_capture(video_path):
    """
    Initialize the video capture object.

    Parameters:
    - video_path (str): Path to the input video file.

    Returns:
    - cap (cv2.VideoCapture): Video capture object.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()
    return cap

def process_frames(cap, fgbg):
    """
    Process each frame of the video, perform background subtraction,
    and accumulate motion information to create a developing heatmap.

    Parameters:
    - cap (cv2.VideoCapture): Video capture object.
    - fgbg: Background subtractor object.

    Returns:
    - first_frame (numpy.ndarray): The first frame of the video.
    - accum_image (numpy.ndarray): Accumulated motion information (heatmap).
    """
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
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fgmask = fgbg.apply(gray)

            thresh = 2
            maxValue = 2
            _, th1 = cv2.threshold(fgmask, thresh, maxValue, cv2.THRESH_BINARY)
            accum_image = cv2.add(accum_image, th1)

            # Display live heatmap
            color_image = cv2.applyColorMap(accum_image, cv2.COLORMAP_HOT)
            live_heatmap = cv2.addWeighted(frame, 0.7, color_image, 0.7, 0)
            cv2.imshow('Live Heatmap', live_heatmap)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    progress_bar.close()
    return first_frame, accum_image

def generate_overlay(first_frame, accum_image):
    """
    Generate the final heatmap overlay by applying color map to the accumulated image.

    Parameters:
    - first_frame (numpy.ndarray): The first frame of the video.
    - accum_image (numpy.ndarray): Accumulated motion information (heatmap).

    Returns:
    - result_overlay (numpy.ndarray): Final heatmap overlay.
    """
    color_image = cv2.applyColorMap(accum_image, cv2.COLORMAP_HOT)
    result_overlay = cv2.addWeighted(first_frame, 0.7, color_image, 0.7, 0)
    return result_overlay

def save_image(output_folder, result_overlay):
    """
    Save the heatmap overlay image, appending a number if the filename already exists.

    Parameters:
    - output_folder (str): Path to the output folder.
    - result_overlay (numpy.ndarray): Final heatmap overlay.

    Returns:
    - output_path (str): Path to the saved heatmap overlay image.
    """
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

    first_frame, accum_image = process_frames(cap, fgbg)
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
