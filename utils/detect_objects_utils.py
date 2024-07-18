import os
import cv2
from datetime import datetime

import tempfile


def img_write(crop_img, crop_images_dir, img_counter, timestamp):
    """
    切り取られた画像を指定されたディレクトリに保存します。

    Args:
        crop_img (numpy.ndarray): 切り取られた画像の配列。
        crop_images_dir (str): 切り取られた画像を保存するディレクトリのパス。
        img_counter (int): 画像の連番カウンター。
        timestamp (float): 画像が切り取られたタイムスタンプ。

    Returns:
        None
    """
    now = datetime.now()
    current_time = now.strftime("%Y%m%d_%H%M%S")
    # filename = f"{img_counter:04d}_{current_time}.png"
    filename = f"{img_counter:04d}_{current_time}_time{timestamp:.2f}s.png"

    if not os.path.exists(crop_images_dir):
        os.mkdir(crop_images_dir)

    cv2.imwrite(os.path.join(crop_images_dir, filename), crop_img)


def get_fixed_size(original_height, original_width):
    """
    元の解像度に基づいて、1280ピクセル以下の場合はそのまま、
    それ以上の場合は1280ピクセルに縮小した固定サイズを計算する。

    Args:
        original_height (int): 元の動画の高さ。
        original_width (int): 元の動画の幅。

    Returns:
        tuple: 縮小後の固定された高さと幅のタプル。
    """
    if original_height > 1280 or original_width > 1280:
        # 縮小率を計算する
        scale = min(1280 / original_height, 1280 / original_width)
        # 縮小後の高さと幅を計算する
        fixed_width = int(original_width * scale)
        fixed_height = int(original_height * scale)
    else:
        # 解像度が1280未満の場合はオリジナルの解像度をそのまま使用する
        fixed_width = original_width
        fixed_height = original_height
    return fixed_width, fixed_height


def calculate_points(
    point_list, original_width, original_height, fixed_width, fixed_height
):
    """
    ポイントリストと固定サイズに基づいてエリアを計算します。

    Args:
        point_list (list): ポイントのリスト。
        original_width (int): 元の動画の幅。
        original_height (int): 元の動画の高さ。
        fixed_width (int): 縮小後の固定された幅。
        fixed_height (int): 縮小後の固定された高さ。

    Returns:
        list: 計算されたエリアの座標リスト。
    """
    area = [
        (
            int(coord[0] * fixed_width / original_width),
            int(coord[1] * fixed_height / original_height),
        )
        for coord in point_list
    ]
    return area




def save_video_to_tempfile(video_data):
    """
    動画データを一時ファイルに保存します。
    """
    with tempfile.NamedTemporaryFile(delete=False) as temp_video_file:
        temp_video_file.write(video_data)
        return temp_video_file.name
