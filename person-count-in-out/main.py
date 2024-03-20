import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import *
import cvzone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

model = YOLO(ROOT_DIR / 'yolov8s.pt')


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)


cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap = cv2.VideoCapture('p.mp4')

with open(ROOT_DIR / "coco.txt", "r") as f:
    data: str = f.read()

class_list: list = data.split("\n")

count = 0
tracker = Tracker()
area1 = [(494, 289), (505, 499), (578, 496), (530, 292)]
area2 = [(548, 290), (600, 496), (637, 493), (574, 288)]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #    count += 1
    #    if count % 3 != 0:
    #        continue
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
    bbox_idx = tracker.update(detect_list)
    for bbox in bbox_idx:
        x3, y3, x4, y4, id1 = bbox
        cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 255, 255), 2)
        cvzone.putTextRect(frame, f'{id1}', (x3, y3), 1, 1)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()

if __name__ == '__main__':
    print('a')
