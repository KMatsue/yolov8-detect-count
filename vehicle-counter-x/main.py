import math
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO
import numpy as np
from trackers.tracker import Tracker

ROOT_DIR = Path(__file__).resolve().parent.parent
# print(f"{str(Path(__file__).parent)}/car-count-x02.mov")

model = YOLO(ROOT_DIR / "yolov8s.pt")

cap = cv2.VideoCapture(f"{str(Path(__file__).parent)}/car-count-x02.mov")


# fps = float(cap.get(cv2.CAP_PROP_FPS))


def mouse_event(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)


cv2.namedWindow("counter_window")
cv2.setMouseCallback("counter_window", mouse_event)

frame_size = (1020, 600)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output_movie_path = f"{str(Path(__file__).parent)}/ImgVideo03.avi"
# # print(fourcc)
video = cv2.VideoWriter(output_movie_path, fourcc, 29, frame_size)

with open(ROOT_DIR / "coco.txt", "r") as f:
    data: str = f.read()

class_list: list = data.split("\n")

count = 0

tracker = Tracker()
# tracker = Tracker2(max_distance=32)

cx1: int = 500
cx2: int = 630
offset: int = 16

vh_down = {}
down_counter = []

vh_up = {}
up_counter = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 2 != 0:
        continue
    frame = cv2.resize(frame, frame_size)

    results = model.predict(frame)
    print(results[0].boxes)
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
        conf = math.ceil(row[4] * 100)
        print(conf)
        class_index = int(row[5])
        detect_classnames = class_list[class_index]
        if "car" or "truck" in detect_classnames:
            detect_list.append([x1, y1, x2, y2])

    bbox_ids = tracker.update(detect_list)

    for bbox in bbox_ids:
        x3, y3, x4, y4, id = bbox

        # オブジェクトの中心座標
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        cv2.rectangle(frame, (x3, y3), (x4, y4), (200, 10, 100), 1)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(
            frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2
        )
        # ----右方向に動く車両のカウント----
        # cyの値がcy1の±offset範囲内の場合True
        if (cx + offset) > cx1 > (cx - offset):
            vh_down[id] = cx
        if id in vh_down:
            # cyの値がcy2の±offset範囲内の場合True
            if (cx + offset) > cx2 > (cx - offset):
                cv2.line(frame, (cx2, 40), (cx2, 520), (0, 0, 255), 2)
                # cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)
                # cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                # cv2.putText(
                #     frame,
                #     str(id),
                #     (x3, y3),
                #     cv2.FONT_HERSHEY_COMPLEX,
                #     0.8,
                #     (0, 255, 255),
                #     2)

                if down_counter.count(id) == 0:
                    down_counter.append(id)

        # ---- 左方向に動く車両のカウント----
        # cyの値がcy2の±offset範囲内の場合True
        if (cx + offset) > cx2 > (cx - offset):
            vh_up[id] = cx
        if id in vh_up:
            # cyの値がcy1の±offset範囲内の場合True
            if (cx + offset) > cx1 > (cx - offset):
                cv2.line(frame, (cx1, 40), (cx1, 520), (0, 0, 255), 2)
                # cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 100), 2)
                # cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                # cv2.putText(
                #     frame,
                #     str(id),
                #     (cx, cy),
                #     cv2.FONT_HERSHEY_COMPLEX,
                #     0.8,
                #     (0, 255, 255),
                #     2,
                # )
                if up_counter.count(id) == 0:
                    up_counter.append(id)

    # line1
    cv2.line(frame, (cx1, 40), (cx1, 520), (255, 255, 255), 1)
    cv2.putText(
        frame,
        f"1line:cx1({cx1})",
        (cx1, 40),
        cv2.FONT_HERSHEY_COMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    # line2
    cv2.line(frame, (cx2, 40), (cx2, 520), (255, 255, 255), 1)
    cv2.putText(
        frame,
        f"2line:cx2({cx2})",
        (cx2, 40),
        cv2.FONT_HERSHEY_COMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    # カウント数表示
    cv2.putText(
        frame,
        f"RightCount:{len(down_counter)}",
        (60, 55),
        cv2.FONT_HERSHEY_COMPLEX,
        0.6,
        (255, 100, 255),
        2,
    )
    cv2.putText(
        frame,
        f"LeftCount:{len(up_counter)}",
        (60, 115),
        cv2.FONT_HERSHEY_COMPLEX,
        0.6,
        (255, 100, 255),
        2,
    )
    print(vh_down)
    print(f"down:{down_counter}")
    print(f"up:{up_counter}")

    cv2.imshow("counter_window", frame)
    # video.write(frame)

    if cv2.waitKey(0) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

if __name__ == "__main__":
    print("a")
    print(np.empty((2, 5)))
