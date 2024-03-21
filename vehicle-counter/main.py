import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import cvzone
from tracker import *
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

model = YOLO(ROOT_DIR / 'yolov8s.pt')

cap = cv2.VideoCapture('car-count-y.mp4')


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)


cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

with open(ROOT_DIR / "coco.txt", "r") as f:
    data: str = f.read()

class_list: list = data.split("\n")
count = 0

tracker = Tracker()

cy1 = 323
cy2 = 368
offset = 6

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    #   print(results)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    #    print(px)
    detect_list = []

    for index, row in px.iterrows():
        #        print(row)

        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
            detect_list.append([x1, y1, x2, y2])
    bbox_ids = tracker.update(detect_list)
    for bbox in bbox_ids:
        x3, y3, x4, y4, id = bbox

        # オブジェクトの中心座標
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2

        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    cv2.line(frame, (264, cy1), (829, cy1), (255, 255, 255), 1)
    cv2.putText(frame, '1line', (270, 318), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 255), 2)
    cv2.line(frame, (167, cy2), (932, cy2), (255, 255, 255), 1)
    cv2.putText(frame, '2line', (177, 360), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(0) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()

if __name__ == '__main__':
    print('a')
