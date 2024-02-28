import numpy as np
import cv2

VIDEO_PATH = r"Path to video file"

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    fgbg = cv2.createBackgroundSubtractorMOG2()

    first_iteration_indicator = True

    while True:
        ret, frame = cap.read()

        if not ret:
            print("End of video.")
            break

        if first_iteration_indicator:
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

            color_image = cv2.applyColorMap(accum_image, cv2.COLORMAP_HOT)
            result_overlay = cv2.addWeighted(frame, 0.7, color_image, 0.7, 0)

            cv2.imshow('Live Heatmap Overlay', result_overlay)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
