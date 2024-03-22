from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

from tracker import Tracker

ROOT_DIR = Path(__file__).resolve().parent.parent

model = YOLO(ROOT_DIR / "yolov8s.pt")

cap = cv2.VideoCapture("car-count-y.mp4")


def mouse_event(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)


cv2.namedWindow("counter_window")
cv2.setMouseCallback("counter_window", mouse_event)

with open(ROOT_DIR / "coco.txt", "r") as f:
    data: str = f.read()

class_list: list = data.split("\n")
count = 0

tracker = Tracker()

cy1: int = 323
cy2: int = 368
offset: int = 6

vh_down = {}
down_counter = []

vh_up = {}
up_counter = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    # print(results)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    # print(px)
    detect_list = []

    for index, row in px.iterrows():
        # print(row)

        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if "car" in c:
            detect_list.append([x1, y1, x2, y2])
    bbox_ids = tracker.update(detect_list)
    for bbox in bbox_ids:
        x3, y3, x4, y4, id = bbox

        # オブジェクトの中心座標
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2

        # ----手前方向に動く車両のカウント----
        # cyの値がcy1の±offset範囲内の場合True
        if (cy + offset) > cy1 > (cy - offset):
            vh_down[id] = cy
        if id in vh_down:
            # cyの値がcy2の±offset範囲内の場合True
            if (cy + offset) > cy2 > (cy - offset):
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    str(id),
                    (cx, cy),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
                if down_counter.count(id) == 0:
                    down_counter.append(id)

        # ----奥方向に動く車両のカウント----
        # cyの値がcy2の±offset範囲内の場合True
        if (cy + offset) > cy2 > (cy - offset):
            vh_up[id] = cy
        if id in vh_up:
            # cyの値がcy1の±offset範囲内の場合True
            if (cy + offset) > cy1 > (cy - offset):
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 100), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    str(id),
                    (cx, cy),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
                if up_counter.count(id) == 0:
                    up_counter.append(id)

    # line1
    cv2.line(frame, (264, cy1), (829, cy1), (255, 255, 255), 1)
    cv2.putText(
        frame, "1line", (270, 318), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 255), 2
    )

    # line2
    cv2.line(frame, (167, cy2), (932, cy2), (255, 255, 255), 1)
    cv2.putText(
        frame, "2line", (177, 360), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 255), 2
    )

    # カウント数表示
    cv2.putText(
        frame,
        f"DownCount:{len(down_counter)}",
        (60, 40),
        cv2.FONT_HERSHEY_COMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"UpCount:{len(up_counter)}",
        (60, 100),
        cv2.FONT_HERSHEY_COMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    print(f"down:{down_counter}")
    print(f"up:{up_counter}")

    cv2.imshow("counter_window", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

if __name__ == "__main__":
    print("a")
