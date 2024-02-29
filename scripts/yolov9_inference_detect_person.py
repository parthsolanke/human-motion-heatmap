import yolov9
import cv2
import numpy as np

# load pretrained or custom model
model = yolov9.load(
    "./yolo-coco/yolov9/gelan-c.pt",
    device="cpu",
)

# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.classes = None  # 0: class id for person

video_path = "./data/inp.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error opening video file")
    exit()

while cap.isOpened():
    
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

    # perform inference on frame
    results = model(frame)

    # parse results
    predictions = results.pred[0]
    boxes = predictions[:, :4]  # x1, y1, x2, y2s

    # draw bounding boxes on frame
    for box in boxes:
        x1, y1, x2, y2 = np.array(box, dtype=int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 1, cv2.LINE_AA)
          
    # display frame
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
