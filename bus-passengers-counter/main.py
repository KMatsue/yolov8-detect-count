import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tracker import *
import cvzone
from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parent.parent

model = YOLO(ROOT_DIR / 'yolov8s.pt')


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)


print("file exists?", os.path.exists('busfinal.mp4'))

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap = cv2.VideoCapture('busfinal.mp4')

with open(ROOT_DIR / "coco.txt", "r") as f:
    data: str = f.read()

class_list: list = data.split("\n")

count = 0

area1 = [(259, 488), (281, 499), (371, 499), (303, 466)]

tracker = Tracker()

counter = []

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

        if 'person' in c:
            detect_list.append([x1, y1, x2, y2])
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

    bbox_idx = tracker.update(detect_list)
    print(type(bbox_idx))
    for object_id, rect in bbox_idx.items():
        x3, y3, x4, y4 = rect
        results = cv2.pointPolygonTest(np.array(area1, np.int32), (x3, y4), False)
        if results >= 0:
            if counter.count(object_id) == 0:
                counter.append(object_id)
        cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)
        cv2.circle(frame, (x3, y4), 6, (0, 255, 0), -1)
        cvzone.putTextRect(frame, f'{object_id}', (x3, y3), 1, 1)
    cvzone.putTextRect(frame, f'person_count:{len(counter)}', (60, 60), 1, 1)
    cv2.polylines(frame, [np.array(area1, np.int32)], True, (255, 255, 255), 2)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
if __name__ == '__main__':
    print('a')
