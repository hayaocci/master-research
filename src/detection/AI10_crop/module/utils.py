import csv
import os
import cv2

MASTER_RESEARCH_DIR = 'C:/workspace/Github/master-research/'

def read_csv(csv_file_path):
    with open(csv_file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader]
    return header, rows

def make_label_array(header: list, rows: list, label: str):

    # ヘッダーのインデックスを取得
    label_index = header.index(label)

    # データセットの配列を作成
    label_array = []
    for row in rows:
        label_array.append(row[label_index])
    return label_array

def get_file_num(save_dir_path):
    """
    ファイル保存先に存在するファイル数を取得し、その数に応じてファイル名を変更して保存する関数
    """

    # ファイル保存先のディレクトリが存在しない場合は作成する
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    # ファイル保存先のディレクトリ内のファイル数を取得
    file_count = sum(os.path.isfile(os.path.join(save_dir_path, name)) for name in os.listdir(save_dir_path))

    return file_count

def resize_image(input_image_path, width=224, height=224):
    # 画像を読み込む
    image = cv2.imread(input_image_path)
    
    # 指定されたサイズにリサイズ
    resized_image = cv2.resize(image, (width, height))
    
    # リサイズされた画像を保存
    # cv2.imwrite(output_image_path, resized_image)
    return resized_image

if __name__ == '__main__':
    header, rows = read_csv(MASTER_RESEARCH_DIR + 'dataset/001/train.csv')
    # print(header)
    print(rows[0][1])
    print(rows[1][1])
    # gx_list = make_label_array(header, rows, 'gx')

    # print(gx_list)
# 定数の定義
ORIGINAL_SIZE = (512, 512)  # 入力画像サイズ
INPUT_SIZE = (224, 224)  # モデルの入力サイズ

INPUT_CHANNEL = 3  # RGBの場合は3

# matplotlibの設定
MAP_COLOR = 'viridis'  # ヒートマップのカラーマップ
