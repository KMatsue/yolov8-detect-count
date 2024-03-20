import cv2
import numpy as np
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
#  カウントするエリア指定
area1 = [(474, 289), (485, 499), (568, 496), (520, 292)]
area2 = [(548, 290), (600, 496), (647, 493), (584, 288)]

# 中へ歩く人のidを格納する
going_in = {}
# 外へ歩く人のidを格納する
going_out = {}

counter_out = []
counter_in = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # count += 1
    # if count % 2 != 0:
    #     continue

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
        x3, y3, x4, y4, id = bbox
        # results: (x4,y4)のポイントがarea内の時:1,エリア外は:-1, エリアの輪郭上の場合は:0,を返す
        results = cv2.pointPolygonTest(np.array(area2, np.int32), (x4, y4), False)
        if results >= 0:
            going_out[id] = (x4, y4)
        if id in going_out:
            results1 = cv2.pointPolygonTest(np.array(area1, np.int32), (x4, y4), False)
            if results1 >= 0:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 255, 255), 2)
                cv2.circle(frame, (x4, y4), 4, (255, 0, 255), -1)
                cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 1)
                if counter_out.count(id) == 0:
                    counter_out.append(id)

        results2 = cv2.pointPolygonTest(np.array(area1, np.int32), (x4, y4), False)
        if results2 >= 0:
            going_in[id] = (x4, y4)
        if id in going_in:
            results3 = cv2.pointPolygonTest(np.array(area2, np.int32), (x4, y4), False)
            if results3 >= 0:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 0), 2)
                cv2.circle(frame, (x4, y4), 4, (255, 0, 255), -1)
                cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 1)
                if counter_in.count(id) == 0:
                    counter_in.append(id)
    print(counter_in)
    # カウント数描画
    cvzone.putTextRect(frame, f'OUT_COUNT:{len(counter_out)}', (50, 60), 2, 2)
    cvzone.putTextRect(frame, f'IN_COUNT:{len(counter_in)}', (50, 160), 2, 2)

    # カウントエリア描画
    cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 255, 0), 1)
    cv2.polylines(frame, [np.array(area2, np.int32)], True, (0, 255, 255), 1)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

if __name__ == '__main__':
    print('a')
