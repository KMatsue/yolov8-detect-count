import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *
from datetime import datetime
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
PARENT_DIR = Path(__file__).resolve().parent

model = YOLO(ROOT_DIR / 'yolov8s.pt')


def rgb(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
    elif event == cv2.EVENT_LBUTTONDOWN:
        click_points = [x, y]

        print(click_points, flags, param)


cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', rgb)

cap = cv2.VideoCapture(str(ROOT_DIR / 'movies/peoplecount.mp4'))
# cap = cv2.VideoCapture(0)

with open(ROOT_DIR / "coco.txt", "r") as f:
    data: str = f.read()

class_list: list = data.split("\n")
# print(class_list)
count: int = 0
tracker = Tracker()
area: list = [(54, 436), (41, 449), (317, 494), (317, 470)]
area_c = set()


def img_write(img):
    now = datetime.now()
    current_time = now.strftime("%d_%m_%Y_%H_%M_%S")
    filename = '%s.png' % current_time

    crop_images_dir = PARENT_DIR / 'crop_images'
    if not os.path.exists(crop_images_dir):
        os.mkdir(crop_images_dir)

    cv2.imwrite(os.path.join(crop_images_dir, filename),
                img)


fps = float(cap.get(cv2.CAP_PROP_FPS))
fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
# fourcc = fourcc_int.to_bytes(4, 'little').decode('utf-8')
# fourcc = list((fourcc_int.to_bytes(4, 'little').decode('utf-8')))
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# fourcc = cv2.fourcc('m', 'p', '4', 'v')
# video = cv2.VideoWriter('../ImgVideo.mp4', fourcc_int, 20, (1020, 500))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1

    # 2フレームに1回処理を早くするためスキップする
    if count % 2 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    #   print(results)

    detect_data = results[0].boxes.data
    # print(detect_data)
    df = pd.DataFrame(detect_data).astype("float")
    # print(df)
    detect_list = []
    for index, row in df.iterrows():
        # print(row)
        # (x1,y1):バウンディングbox左上座標, (x2,y2):BBox右下座標, d:検出種別(例:0=person)
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
        results = cv2.pointPolygonTest(np.array(area, np.int32), (x4, y4), False)
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 1)
        cv2.circle(frame, (x4, y4), 4, (255, 0, 255), -1)
        cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
        if results >= 0:
            crop = frame[y3:y4, x3:x4]
            img_write(crop)
            #            cv2.imshow(str(id),crop)
            area_c.add(id)

    cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 0, 0), 1)
    # --- カウント表示 ---
    # print(f'area_c:{area_c}')
    k = len(area_c)
    cv2.putText(frame, str(k), (50, 60), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 3)

    cv2.imshow("RGB", frame)

    # 動画の解像度が異なると保存できないので確認する
    # assert frame.shape[:2][::-1] == (1020, 500)
    # video.write(frame)

    # 0xFF == 27(escキー)が押されると1ms後に画面が閉じる
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()

if __name__ == "__main__":
    print("a")
