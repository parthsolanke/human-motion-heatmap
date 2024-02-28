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
model.classes = None  # (optional list) filter by class

video_path = "./data/input.mp4"
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
    boxes = predictions[:, :4]  # x1, y1, x2, y2

    # add missing variables
    scores = predictions[:, 4]  # scores
    categories = predictions[:, 5]  # categories

    # draw bounding boxes on frame
    for box, score, category in zip(boxes, scores, categories):
        x1, y1, x2, y2 = np.array(box, dtype=int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"{score:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 165, 255),
            1,
            cv2.LINE_AA
        )
        category_name = model.names[int(category)]
        cv2.putText(
            frame,
            category_name,
            (x1, y1 - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 165, 255),
            1,
            cv2.LINE_AA
        )
          
    # display frame
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
