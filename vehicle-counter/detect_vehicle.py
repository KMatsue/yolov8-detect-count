import ast
from pathlib import Path

import cv2
import math
import pandas as pd
from ultralytics import YOLO

from trackers.histgram_tracker import HistogramTracker
from utils.detect_objects_utils import get_fixed_size, calculate_points

ROOT_DIR = Path(__file__).resolve().parent.parent
PARENT_DIR = Path(__file__).resolve().parent


def mouse_event(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)


cv2.namedWindow("counter_window")
cv2.setMouseCallback("counter_window", mouse_event)


def detect_vehicle(
        video,
        line1_points,
        line2_points,
        distance,
        confidence,
        direction,
        # is_stopped,
        # progress_callback,
        # frame_callback=None,
):
    """
    指定された動画で車両を検出し、指定されたラインを横切る車両の移動をカウントします。

    Parameters:
    video (Video): 動画ファイルを含むVideoオブジェクト。
    line1_points (str): 最初のラインを定義するポイント。
    line2_points (str): 2番目のラインを定義するポイント。
    distance (str): トラッキングの最大距離閾値。
    confidence (str): 信頼度の閾値。
    direction (str): 車両の移動方向、"horizontal"（水平）または "vertical"（垂直）。
    is_stopped (callable): 解析を中断すべきかどうかを判断するためのコールバック関数。
    progress_callback (callable): 進捗状況を送信するためのコールバック関数。
    frame_callback (callable): 解析後のフレームを送信コールバック関数。

    Returns:
    tuple: 解析済み動画のパスと検出されたオブジェクトの総数。

    Raises:
    Exception: 動画解析中にエラーが発生した場合。

    Note:
    この関数は非同期で実行され、定期的に is_stopped 関数をチェックして
    解析プロセスを中断すべきかどうかを判断します。中断された場合、
    その時点までの結果が返されます。また、progress_callback を通じて
    進捗状況を報告します。
    """
    video_title = video.title

    line1_points_list = ast.literal_eval(line1_points)
    line2_points_list = ast.literal_eval(line2_points)
    tracker_distance = int(distance)
    confidence_threshold = int(confidence)

    model = YOLO(ROOT_DIR / "yolov8s.pt")

    with open(ROOT_DIR / "coco.txt", "r") as f:
        class_names = f.read().split("\n")

    analyzed_video_path = str(PARENT_DIR / f"analyzed_{video_title}.mp4")

    # 検出対象のクラスを定義
    target_classes = ["car", "truck"]
    # フレームスキップの間隔を設定
    frame_skip_interval = 1

    # tracker = Tracker(max_distance=tracker_distance)
    tracker = HistogramTracker(max_distance=tracker_distance)

    try:

        # 動画の解像度を取得
        cap = cv2.VideoCapture(str(PARENT_DIR / video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        _, frame = cap.read()
        original_height, original_width, _ = frame.shape

        # 解像度が1280以上の場合はサイズを縮小して固定の高さと幅を決定する
        fixed_width, fixed_height = get_fixed_size(original_height, original_width)

        # 検出エリアの位置を動的に計算
        line2_fixed = calculate_points(
            line1_points_list,
            original_width,
            original_height,
            fixed_width,
            fixed_height,
        )
        line1_fixed = calculate_points(
            line2_points_list,
            original_width,
            original_height,
            fixed_width,
            fixed_height,
        )

        # 固定の高さと幅
        frame_size = (fixed_width, fixed_height)

        fps = float(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*"avc1")

        video_writer = cv2.VideoWriter(
            analyzed_video_path, fourcc, fps / frame_skip_interval, frame_size
        )

        offset: int = 20 if direction == "horizontal" else 10
        # tracker_distance: int = 65 if direction == "horizontal" else 45

        print(
            f"direction:{direction}, offset:{offset}, distance:{tracker_distance}, confidence:{confidence_threshold}"
        )

        frame_count: int = 0
        vehicle_moving_1 = {}
        counter_1 = []
        vehicle_moving_2 = {}
        counter_2 = []
        last_reported_progress = -1

        while True:
            # if is_stopped():
            #     print("解析が中断されました")
            #     break
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % frame_skip_interval != 0:
                continue
            frame = cv2.resize(frame, frame_size)

            # YOLOでフレーム内のオブジェクトを予測
            results = model.predict(frame)
            # print(results[0].boxes)
            detect_data = results[0].boxes.data
            df = pd.DataFrame(detect_data).astype("float")
            # print(px)
            detect_list = []

            for index, row in df.iterrows():

                x1, y1, x2, y2 = map(int, row[:4])
                class_id = int(row[5])
                detect_classname = class_names[class_id]
                conf = math.ceil(row[4] * 100)
                # print(conf)

                if detect_classname in target_classes and conf > confidence_threshold:
                    # print(f"detect_classname:{detect_classname}")
                    detect_list.append([x1, y1, x2, y2])

            bbox_ids = tracker.update(frame, detect_list)

            for obj_id, bbox in bbox_ids.items():
                x3, y3, x4, y4 = bbox

                # オブジェクトの中心座標
                cx = int(x3 + x4) // 2
                cy = int(y3 + y4) // 2

                cv2.rectangle(frame, (x3, y3), (x4, y4), (200, 150, 100), 1)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    str(obj_id),
                    (x3, y3),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

                # 車両の移動方向に応じて座標を設定
                if direction == "vertical":
                    # 垂直方向の移動の場合、y座標を使用
                    vh_center = cy
                    line1_position = line1_fixed[0][1]
                    line2_position = line2_fixed[0][1]
                else:
                    # 水平方向の移動の場合、x座標を使用
                    vh_center = cx
                    line1_position = line1_fixed[0][0]
                    line2_position = line2_fixed[0][0]

                # ----line1からline2方向に動く車両のカウント----
                # vh_centerの値がline1_positionの±offset範囲内の場合True
                if (vh_center + offset) > line1_position > (vh_center - offset):
                    vehicle_moving_1[obj_id] = vh_center
                if obj_id in vehicle_moving_1:
                    # vh_centerの値がline2_positionの±offset範囲内の場合True
                    if (vh_center + offset) > line2_position > (vh_center - offset):
                        cv2.line(frame, line2_fixed[0], line2_fixed[1], (0, 0, 255), 2)

                        if counter_1.count(obj_id) == 0:
                            counter_1.append(obj_id)

                # ---- line2からline1方向に動く車両のカウント----
                # vh_centerの値がline2_positionの±offset範囲内の場合True
                if (vh_center + offset) > line2_position > (vh_center - offset):
                    vehicle_moving_2[obj_id] = vh_center
                if obj_id in vehicle_moving_2:
                    # vh_centerの値がline1_positionの±offset範囲内の場合True
                    if (vh_center + offset) > line1_position > (vh_center - offset):
                        cv2.line(frame, line1_fixed[0], line1_fixed[1], (0, 0, 255), 2)

                        if counter_2.count(obj_id) == 0:
                            counter_2.append(obj_id)

            # line1
            cv2.line(frame, line1_fixed[0], line1_fixed[1], (255, 255, 255), 1)
            cv2.putText(
                frame,
                "line1",
                line1_fixed[0],
                cv2.FONT_HERSHEY_COMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            # line2
            cv2.line(frame, line2_fixed[0], line2_fixed[1], (255, 255, 255), 1)
            cv2.putText(
                frame,
                "line2",
                line2_fixed[0],
                cv2.FONT_HERSHEY_COMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            # カウント数表示
            # line1からline2方向
            cv2.putText(
                frame,
                f"Count1:{len(counter_1)}",
                (60, 55),
                cv2.FONT_HERSHEY_COMPLEX,
                0.6,
                (255, 100, 255),
                2,
            )
            # line2からline1方向
            cv2.putText(
                frame,
                f"Count2:{len(counter_2)}",
                (60, 115),
                cv2.FONT_HERSHEY_COMPLEX,
                0.6,
                (255, 100, 255),
                2,
            )

            # 進捗状況を表示
            progress = (frame_count / total_frames) * 100
            current_progress = int(progress)
            # 1%単位で進捗を報告
            if current_progress > last_reported_progress:
                # await progress_callback(current_progress)
                last_reported_progress = current_progress

            # if frame_callback:
            #     _, jpeg = cv2.imencode(".jpg", frame)
            #     await frame_callback(jpeg.tobytes())

            print(f"Progress: {progress:.2f}%")

            print(f"line1 to line2:{counter_1}")
            print(f"line2 to line1:{counter_2}")
            cv2.imshow("counter_window", frame)
            if cv2.waitKey(0) & 0xFF == 27:
                break

            video_writer.write(frame)
            object_count = len(counter_1) + len(counter_2)

        cap.release()
        video_writer.release()

        print("動画解析終了!")
        return analyzed_video_path, object_count
    except Exception as e:
        print(f"動画解析中にエラーが発生しました: {e}")
        raise
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    line1_points = "[(405, 932),(797, 907)]"
    line2_points = "[(834, 948),(367, 975)]"
    direction = "horizontal"
    # direction = 'vertical'
    video_path = 'car-count-y.mp4'

    detect_vehicle(video_path, line1_points, line2_points, distance=80, confidence=50, direction=direction)
