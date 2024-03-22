import math


class Tracker:
    def __init__(self):
        # オブジェクトの中心座標を格納する
        self.center_points = {}
        # IDのカウントを保持する
        # 新しいオブジェクトのIDが検出されるたびに、カウントが1ずつ増える
        self.id_count = 0

    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        #  新しいオブジェクトの中心点を取得
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # そのオブジェクトがすでに検出されているかどうかを調べる
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 35:
                    self.center_points[id] = (cx, cy)
                    #                    print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # 新しいオブジェクトが検出されたら、そのオブジェクトにIDを割り当てる
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # もう使われていないIDSを削除するために、中心点によって辞書をきれいにする
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # 使用されていないIDを削除して辞書を更新
        self.center_points = new_center_points.copy()
        return objects_bbs_ids
