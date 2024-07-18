import cv2


class HistogramTracker:
    def __init__(self, max_distance=50, hist_threshold=0.5):
        self.tracked_objects = {}  # 追跡オブジェクトをIDとともに格納する辞書
        self.max_distance = max_distance  # 一致するオブジェクトの最大距離
        self.hist_threshold = hist_threshold  # ヒストグラム類似度の閾値
        self.next_object_id = 1  # 物体にユニークなIDを割り当てるためのカウンタ

    def calculate_histogram(self, frame, rect):
        # 指定された矩形（rect）の領域からカラーヒストグラムを計算する
        x1, y1, x2, y2 = rect
        roi = frame[y1:y2, x1:x2]  # 矩形領域を抽出
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  # HSV色空間に変換
        hist = cv2.calcHist(
            [hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256]
        )  # ヒストグラムを計算
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)  # ヒストグラムを正規化
        return hist

    def update(self, frame, new_rectangles):
        updated_objects = {}  # 更新されたオブジェクトを格納する辞書を初期化

        for new_rect in new_rectangles:
            matched = False  # 新しい矩形が既存のオブジェクトと一致したかを示すフラグ
            new_center = (
                (new_rect[0] + new_rect[2]) / 2,
                (new_rect[1] + new_rect[3]) / 2,
            )  # 新しい矩形の中心を計算
            new_hist = self.calculate_histogram(
                frame, new_rect
            )  # 新しい矩形のヒストグラムを計算

            for obj_id, (obj_rect, obj_hist) in self.tracked_objects.items():
                obj_center = (
                    (obj_rect[0] + obj_rect[2]) / 2,
                    (obj_rect[1] + obj_rect[3]) / 2,
                )  # 既存のオブジェクトの中心を計算
                distance = (
                                   (new_center[0] - obj_center[0]) ** 2
                                   + (new_center[1] - obj_center[1]) ** 2
                           ) ** 0.5  # 中心間のユークリッド距離を計算
                hist_dist = cv2.compareHist(
                    obj_hist, new_hist, cv2.HISTCMP_BHATTACHARYYA
                )  # ヒストグラムの類似度を計算

                # 距離とヒストグラム類似度の閾値を満たす場合、同一オブジェクトと見なす
                if distance <= self.max_distance and hist_dist < self.hist_threshold:
                    updated_objects[obj_id] = (new_rect, new_hist)  # オブジェクトを更新
                    matched = True  # 一致フラグを設定
                    break

            # 一致しない場合、新しいオブジェクトとして追加
            if not matched:
                updated_objects[self.next_object_id] = (new_rect, new_hist)
                self.next_object_id += 1  # 次のオブジェクトIDをインクリメント

        self.tracked_objects = updated_objects  # 更新されたオブジェクトで辞書を更新
        return {
            obj_id: obj_rect for obj_id, (obj_rect, obj_hist) in updated_objects.items()
        }  # オブジェクトIDと矩形を返す
