import os
import cv2
import numpy as np
import module.basic_func as bf
from tqdm import tqdm
import time

def make_seg_label(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ディレクトリ内のファイル数を取得
    file_count = bf.get_file_num(input_dir)

    t_start = time.time()

    # ファイル数分ループ
    for i in tqdm(range(file_count)):
        # 画像のパスを取得
        image_path = os.path.join(input_dir, str(i) + '.png')
        label_img = bf.make_seg_label(image_path)
        output_path = os.path.join(output_dir, str(i) + '.png')
        cv2.imwrite(output_path, label_img)

    t_end = time.time()
    print(f'elapsed time: {t_end - t_start} [s]')

if __name__ == '__main__':
    input_dir = '../dataset/003/train/output'
    output_dir = '../dataset/003/train/seg_label'
    make_seg_label(input_dir, output_dir)

    input_dir = '../dataset/003/valid/output'
    output_dir = '../dataset/003/valid/seg_label'
    make_seg_label(input_dir, output_dir)

