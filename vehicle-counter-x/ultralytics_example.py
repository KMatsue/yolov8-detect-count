# 公式(ultralytics)の例
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO
from ultralytics.solutions import object_counter

ROOT_DIR = Path(__file__).resolve().parent.parent

model = YOLO(ROOT_DIR / "yolov8s.pt")


def mouse_event(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)


cap = cv2.VideoCapture(f"{str(Path(__file__).parent)}/car-count-x02.mov")

cv2.namedWindow("counter_window")
cv2.setMouseCallback("counter_window", mouse_event)

assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# line_points = [(20, 400), (1080, 400)]  # line or region points
line_points = [(1020 // 2, 10), (1020 // 2, 490)]
classes_to_count = [0, 2]  # person and car classes for count

# Video writer
video_writer = cv2.VideoWriter("object_counting_output.avi",
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               fps,
                               (w, h))

# Init Object Counter
counter = object_counter.ObjectCounter()
counter.set_args(view_img=True,
                 reg_pts=line_points,
                 classes_names=model.names,
                 draw_tracks=True)
frame_size = (1020, 500)
while cap.isOpened():
    # print(w, h, line_points)
    success, im0 = cap.read()
    im0 = cv2.resize(im0, frame_size)
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    tracks = model.track(im0, persist=True, show=False,
                         classes=classes_to_count)
    print(tracks)
    a = tracks[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    print(px)
    im0 = counter.start_counting(im0, tracks)
    # cv2.imshow("counter_window", im0)
    video_writer.write(im0)

    if cv2.waitKey(0 ) & 0xFF == 27:
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()

if __name__ == "__main__":
    print("a")
