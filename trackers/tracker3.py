class Tracker:
    def __init__(self, max_distance=50):
        self.tracked_objects = {}  # 追跡オブジェクトをIDとともに格納する辞書
        self.max_distance = max_distance  # 一致するオブジェクトの最大距離
        self.next_object_id = 1  # 物体にユニークなIDを割り当てるためのカウンタ

    def update(self, new_rectangles):
        # 更新されたオブジェクトを格納する辞書を初期化する
        updated_objects = {}

        # 検出された新しい矩形を反復処理する
        for new_rect in new_rectangles:
            matched = False

            # 既存の追跡対象オブジェクトを反復処理する
            for obj_id, obj_rect in self.tracked_objects.items():

                # 新しい矩形の中心を計算する
                new_center = (
                    (new_rect[0] + new_rect[2]) / 2,
                    (new_rect[1] + new_rect[3]) / 2,
                )

                # 既存の追跡対象オブジェクトの矩形の中心を計算する
                obj_center = (
                    (obj_rect[0] + obj_rect[2]) / 2,
                    (obj_rect[1] + obj_rect[3]) / 2,
                )

                # 中心間のユークリッド距離を計算する
                distance = (
                    (new_center[0] - obj_center[0]) ** 2
                    + (new_center[1] - obj_center[1]) ** 2
                ) ** 0.5

                # 距離が閾値以内であれば、追跡対象オブジェクトを更新する
                if distance <= self.max_distance:
                    updated_objects[obj_id] = new_rect
                    matched = True
                    break

            # 一致しない場合は、新しい追跡対象オブジェクトを作成する
            if not matched:
                updated_objects[self.next_object_id] = new_rect
                self.next_object_id += 1  # Increment the object ID

        # 更新されたオブジェクトでtracked_objects辞書を更新する
        self.tracked_objects = updated_objects
        # 更新されたオブジェクトをIDで返す
        return self.tracked_objects
