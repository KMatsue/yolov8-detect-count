import ast
from pathlib import Path

import cv2
import math
import numpy as np
import pandas as pd
from ultralytics import YOLO

from trackers.tracker3 import Tracker
from utils.detect_objects_utils import img_write, get_fixed_size, calculate_points

ROOT_DIR = Path(__file__).resolve().parent.parent
PARENT_DIR = Path(__file__).resolve().parent


def rgb(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
    elif event == cv2.EVENT_LBUTTONDOWN:
        click_points = [x, y]

        print(click_points, flags, param)


cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', rgb)


def detect_person(
        video,
        points,
        distance,
        confidence,
):
    """
    指定された動画で人物を検出し、指定されたエリア内で検出された人物の画像を保存します。

    Parameters:
    video (Video): 動画ファイルを含むVideoオブジェクト。
    points (str): エリアを定義するポイントのリスト。
    distance (str): トラッキングの最大距離閾値。
    confidence (str): 信頼度の閾値。

    Returns:
    tuple: 解析済み動画のパス、保存された画像のディレクトリ、検出されたオブジェクトの総数。

    Raises:
    Exception: 動画解析中にエラーが発生した場合。

    Note:
    この関数は非同期で実行され、定期的に is_stopped 関数をチェックして
    解析プロセスを中断すべきかどうかを判断します。中断された場合、
    その時点までの結果が返されます。また、progress_callback を通じて
    進捗状況を報告します。
    """

    video_title = video.title
    point_list = ast.literal_eval(points)
    tracker_distance = int(distance)
    confidence_threshold = int(confidence)

    model = YOLO(ROOT_DIR / "yolov8s.pt")

    with open(ROOT_DIR / "coco.txt", "r") as f:
        class_names = f.read().split("\n")

    analyzed_video_path = str(PARENT_DIR / f"analyzed_{video_title}.mp4")

    # 検出対象のクラスを定義
    target_classes = ["person"]
    # フレームスキップの間隔を設定
    frame_skip_interval = 2

    tracker = Tracker(max_distance=tracker_distance)

    try:

        # 動画の解像度を取得
        cap = cv2.VideoCapture(str(ROOT_DIR / video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        _, frame = cap.read()
        original_height, original_width, _ = frame.shape

        # 解像度が1280以上の場合はサイズを縮小して固定の高さと幅を決定する
        fixed_width, fixed_height = get_fixed_size(original_height, original_width)

        # 検出エリアの位置を動的に計算
        area = calculate_points(
            point_list, original_width, original_height, fixed_width, fixed_height
        )

        # 固定の高さと幅
        frame_size = (fixed_width, fixed_height)

        fps = float(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*"avc1")

        video_writer = cv2.VideoWriter(
            analyzed_video_path, fourcc, fps / frame_skip_interval, frame_size
        )

        crop_images_dir = PARENT_DIR / "crop_images"

        detected_ids = set()
        frame_count: int = 0
        img_counter: int = 0
        last_reported_progress = -1

        while True:

            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # 2フレームに1回処理を早くするためスキップする
            if frame_count % frame_skip_interval != 0:
                continue

            frame = cv2.resize(frame, frame_size)

            # オブジェクト切り取りようのフレーム
            crop_frame = cv2.resize(frame.copy(), frame_size)

            results = model.predict(frame)

            detect_data = results[0].boxes.data
            df = pd.DataFrame(detect_data).astype("float")
            # print(df)
            detect_list = []
            for index, row in df.iterrows():
                # print(row)
                # (x1,y1):バウンディングbox左上座標, (x2,y2):BBox右下座標, d:検出種別(例:0=person)
                x1, y1, x2, y2 = map(int, row[:4])
                class_id = int(row[5])
                class_name = class_names[class_id]
                conf = math.ceil(row[4] * 100)
                if class_name in target_classes and conf > confidence_threshold:
                    detect_list.append([x1, y1, x2, y2])

            bbox_idx = tracker.update(detect_list)

            for obj_id, bbox in bbox_idx.items():
                x3, y3, x4, y4 = bbox
                # results: (x4,y4)のポイントがarea内の時:1,エリア外は:-1, エリアの輪郭上の場合は:0,を返す
                results = cv2.pointPolygonTest(
                    np.array(area, np.int32), (x4, y4), False
                )

                if results >= 0:
                    if obj_id not in detected_ids:
                        crop = crop_frame[y3:y4, x3:x4]
                        img_counter += 1
                        timestamp = frame_count / fps
                        img_write(crop, crop_images_dir, img_counter, timestamp)
                        detected_ids.add(obj_id)

                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 1)
                cv2.circle(frame, (x4, y4), 4, (255, 0, 255), -1)
                cv2.putText(
                    frame,
                    str(obj_id),
                    (x3, y3),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                )

            cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 0, 0), 1)

            # --- カウント表示 ---
            object_count = len(detected_ids)
            cv2.putText(
                frame,
                f"Count:{object_count}",
                (50, 80),
                cv2.FONT_HERSHEY_COMPLEX,
                2,
                (255, 0, 0),
                2,
            )

            # 進捗状況を表示
            progress = (frame_count / total_frames) * 100
            current_progress = int(progress)
            # 1%単位で進捗を報告
            if current_progress > last_reported_progress:
                last_reported_progress = current_progress

            print(f"Progress: {progress:.2f}%")
            cv2.imshow("RGB", frame)

            # 0xFF == 27(escキー)が押されると1ms後に画面が閉じる
            if cv2.waitKey(1) & 0xFF == 27:
                break

            video_writer.write(frame)

        cap.release()
        video_writer.release()
        print("動画解析終了!")
        return analyzed_video_path, crop_images_dir, object_count
    except Exception as e:
        print(f"動画解析中にエラーが発生しました: {e}")
        raise
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # points = "[(54, 436), (41, 449), (317, 494), (317, 470)]"
    points = "[(405, 932),(797, 907),(834, 948),(367, 975)]"
    video_path = 'movies/peoplecount.mp4'
    detect_person(video_path, points=points, distance=50, confidence=50)
