# データセット作成用のスクリプト

# インポート
import bpy
import os
import csv
import math
# from utils import *

DATASET_MAIN_PATH = 'C:/workspace/Github/master-research/dataset'


def get_last_number_in_directory(dir_path: str) -> int:
    """
    指定したディレクトリ直下にある、数字のみの名前からなるサブディレクトリのうち
    最大の数値を返します。該当するサブディレクトリがない場合は None を返します。
    """
    candidates = []
    for item in os.listdir(dir_path):
        full_path = os.path.join(dir_path, item)
        # ディレクトリかつ数字の名前であるかを判定
        if os.path.isdir(full_path) and item.isdigit():
            candidates.append(int(item))

    return max(candidates) if candidates else None

# def make_dataset_dir(dir_num):
#     """
#     データセット用のディレクトリを作成します。
#     """
#     # データセット用のディレクトリを作成
#     os.makedirs(DATASET_PATH, exist_ok=True)
#     # 画像用のディレクトリを作成
#     os.makedirs(os.path.join(DATASET_PATH, 'images'), exist_ok=True)
#     # マスク用のディレクトリを作成
#     os.makedirs(os.path.join(DATASET_PATH, 'masks'), exist_ok=True)
#     # マスクのラベル用のディレクトリを作成
#     os.makedirs(os.path.join(DATASET_PATH, 'labels'), exist_ok=True)
#     # ディレクトリ番号を記録
#     with open(os.path.join(DATASET_PATH, 'dir_num.txt'), 'w') as f:
#         f.write(str(dir_num))

def rotate_object_deg(object: str, deg_list: list):
    obj = bpy.data.objects[object]

    # degreeのリストからradianのリストを作成
    rad_list = [math.radians(deg) for deg in deg_list]

    # x, y, zそれぞれの軸周りに回転
    for i, rad in enumerate(rad_list):
        obj.rotation_euler[i] = rad

def main(dataset_main_path: str):
    """
    データセットを作成するメイン関数
    """
    
    # メインディレクトリ内の番号調査
    # last_num = get_last_number_in_directory(dataset_main_path)
    # if last_num is not None:
    #     # print(f"最後の番号は {last_num} です。")
    #     dataset_main_path = os.path.join(dataset_main_path, str(last_num + 1))
    # else:
    #     # print("数字のみで構成されたサブディレクトリがありません。")
    #     dataset_main_path = os.path.join(dataset_main_path, '1')

    # データセット用のディレクトリを作成
    os.makedirs(dataset_main_path, exist_ok=True)

    # .pngファイルのみを取得
    file_names = [f for f in os.listdir(dataset_main_path) if f.endswith(".png")]

    first_numbers = []
    for name in file_names:
        # ファイル名を"_"で分割し、最初の部分を取得
        parts = name.split("_")
        if parts and parts[0].isdigit():
            first_numbers.append(int(parts[0]))

    if first_numbers:
        max_first_number = max(first_numbers)
        # print("一番先頭の数値の最大値:", max_first_number)
    else:
        # print("条件に合致するファイルが見つかりませんでした。")
        max_first_number = 0


    # データセット用のディレクトリを作成
    # make_dataset_dir(dataset_main_path)

    # カメラの設定
    bpy.ops.object.camera_add(location=(25, 0, 0))
    # bpy.context.object.rotation_euler[0] = math.radians(90)
    # bpy.context.object.rotation_euler[2] = math.radians(90)
    camera = bpy.data.objects['Rendering Camera']


    # ロールのリスト -180~180度 5度刻み
    # 


    # roll_list = list(range(-180, 180, 30))
    # pitch_list = list(range(-180, 180, 30))
    # yaw_list = list(range(-180, 180, 30))

    roll_list = list(range(max_first_number, 360, 5))
    pitch_list = list(range(0, 360, 5))
    # yaw_list = list(range(0, 360, 5))

    # roll = 0
    # pitch = 0
    yaw = 0
    for roll in roll_list:
        for pitch in pitch_list:
        # for yaw in yaw_list:
            rotate_object_deg('debris_body', [roll, pitch, yaw])
            # レンダリング
            bpy.ops.render.render(write_still=True)

            # レンダリング画像の保存
            bpy.data.images['Render Result'].save_render(filepath=os.path.join(dataset_main_path, f'{roll}_{pitch}_{yaw}.png'))



    # # 条件を変えて繰り返しレンダリング
    # for pic_num in range(10):
 
    #     # オブジェクトの回転
    #     rotate_object_deg('debris_body', [0, 0, pic_num*5])

    #     # レンダリング
    #     bpy.ops.render.render(write_still=True)

    #     # レンダリング画像の保存
    #     bpy.data.images['Render Result'].save_render(filepath=os.path.join(dataset_main_path, 'images', f'{pic_num}.png'))



    # # レンダリング
    # bpy.ops.render.render(write_still=True)

    # # レンダリング画像の保存
    # bpy.data.images['Render Result'].save_render(filepath=os.path.join(dataset_main_path, 'images', '0.png'))

if __name__ == '__main__':

    # データセットのメインディレクトリ ※絶対パスで指定
    DATASET_MAIN_PATH = 'C:/workspace/Github/master-research/dataset/12'

    main(DATASET_MAIN_PATH)

