# import module.basic_func as bf
# import os
# import csv
# from tqdm import tqdm


# def main(dataset_dir):
#     """
#     画像から重心、bboxを取得し、csvに保存する
    
#     """
#     # データセットのディレクトリ
#     train_input_dir = dataset_dir + '/train/input'
#     train_output_dir = dataset_dir + '/train/output'
#     valid_input_dir = dataset_dir + '/valid/input'
#     valid_output_dir = dataset_dir + '/valid/output'

#     # ディレクトリに含まれる画像ファイルの数を取得
#     train_img_count = bf.get_file_num(train_output_dir)
#     valid_img_count = bf.get_file_num(valid_output_dir)

#     # 画像ファイルの数だけ繰り返す
#     for i in range (train_img_count):

#         # 画像ファイルのパスを取得
#         input_image_path = train_output_dir + '/' + str(i) + '.png'
#         # output_image_path = train_output_dir + '/' + str(i) + '.png'

#         # 画像から重心、bboxを取得
#         gX, gY = bf.calculate_centroid(input_image_path)
#         bbox, norm_bbox = bf.calculate_bbox_and_draw(input_image_path)

#         # 重心、bboxをcsvに保存
#         with open(train_output_dir + '/train.csv', 'a') as f:
#             writer = csv.writer(f)
#             writer.writerow([str(i), gX, gY, bbox, norm_bbox])

#     for i in range (valid_img_count):
            
#             # 画像ファイルのパスを取得
#             input_image_path = valid_output_dir + '/' + str(i) + '.png'
#             # output_image_path = valid_output_dir + '/' + str(i) + '.png'
    
#             # 画像から重心、bboxを取得
#             gX, gY = bf.calculate_centroid(input_image_path)
#             bbox, norm_bbox = bf.calculate_bbox_and_draw(input_image_path)
    
#             # 重心、bboxをcsvに保存
#             with open(valid_output_dir + '/valid.csv', 'a') as f:
#                 writer = csv.writer(f)
#                 writer.writerow([str(i), gX, gY, bbox, norm_bbox])

# if __name__ == '__main__':
#     dataset_dir = "C:/workspace/MasterResearch/blender_dataset/20240812_2_512x512"
#     main(dataset_dir)

import module.basic_func as bf
import os
import csv
from tqdm import tqdm


def main(dataset_dir):
    """
    画像から重心、bboxを取得し、csvに保存する
    """
    # データセットのディレクトリ
    train_input_dir = dataset_dir + '/train/input'
    train_output_dir = dataset_dir + '/train/output'
    valid_input_dir = dataset_dir + '/valid/input'
    valid_output_dir = dataset_dir + '/valid/output'

    # ディレクトリに含まれる画像ファイルの数を取得
    train_img_count = bf.get_file_num(train_output_dir)
    valid_img_count = bf.get_file_num(valid_output_dir)

    # 画像ファイルの数だけ繰り返す
    # tqdmを使って進行状況を表示
    with open(dataset_dir + '/train.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Index', 'gx', 'gy', 'cx', 'cy', 'w', 'h', 'norm_cx', 'norm_cy', 'norm_w', 'norm_h'])
        
        for i in tqdm(range(train_img_count), desc='Processing Training Images'):
            # 画像ファイルのパスを取得
            input_image_path = train_output_dir + '/' + str(i) + '.png'

            # 画像から重心、bboxを取得
            gX, gY = bf.calculate_centroid(input_image_path)
            # bbox, norm_bbox = bf.calculate_bbox_and_draw(input_image_path)
            cx, cy, w, h, norm_cx, norm_cy, norm_w, norm_h  = bf.calculate_bbox_and_draw(input_image_path)

            # 重心、bboxをcsvに保存
            writer.writerow([i, gX, gY, cx, cy, w, h, norm_cx, norm_cy, norm_w, norm_h])

    with open(dataset_dir + '/valid.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Index', 'gx', 'gy', 'cx', 'cy', 'w', 'h', 'norm_cx', 'norm_cy', 'norm_w', 'norm_h'])

        for i in tqdm(range(valid_img_count), desc='Processing Validation Images'):
            # 画像ファイルのパスを取得
            input_image_path = valid_output_dir + '/' + str(i) + '.png'

            # 画像から重心、bboxを取得
            gX, gY = bf.calculate_centroid(input_image_path)
            cx, cy, w, h, norm_cx, norm_cy, norm_w, norm_h  = bf.calculate_bbox_and_draw(input_image_path)

            # 重心、bboxをcsvに保存
            writer.writerow([i, gX, gY,cx, cy, w, h, norm_cx, norm_cy, norm_w, norm_h])

def main2(dataset_dir):
    """
    画像から重心、bboxを取得し、csvに保存する
    """
    # データセットのディレクトリ
    train_input_dir = dataset_dir + '/train/input'
    train_output_dir = dataset_dir + '/train/output'
    valid_input_dir = dataset_dir + '/valid/input'
    valid_output_dir = dataset_dir + '/valid/output'

    # ディレクトリに含まれる画像ファイルの数を取得
    train_img_count = bf.get_file_num(train_output_dir)
    valid_img_count = bf.get_file_num(valid_output_dir)

    # 画像ファイルの数だけ繰り返す
    # tqdmを使って進行状況を表示
    with open(dataset_dir + '/train_2.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Index', 'x1', 'y1', 'x2', 'y2', 'norm_x1', 'norm_y1', 'norm_x2', 'norm_y2', 'img_height', 'img_width'])
        
        for i in tqdm(range(train_img_count), desc='Processing Training Images'):
            # 画像ファイルのパスを取得
            input_image_path = train_output_dir + '/' + str(i) + '.png'

            x1, y1, x2, y2, norm_x1, norm_y1, norm_x2, norm_y2, img_height, img_width  = bf.calculate_bbox_and_draw2(input_image_path)

            # 重心、bboxをcsvに保存
            writer.writerow([i, x1, y1, x2, y2, norm_x1, norm_y1, norm_x2, norm_y2, img_height, img_width])

    with open(dataset_dir + '/valid_2.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Index', 'x1', 'y1', 'x2', 'y2', 'norm_x1', 'norm_y1', 'norm_x2', 'norm_y2', 'img_height', 'img_width'])

        for i in tqdm(range(valid_img_count), desc='Processing Validation Images'):
            # 画像ファイルのパスを取得
            input_image_path = valid_output_dir + '/' + str(i) + '.png'

            x1, y1, x2, y2, norm_x1, norm_y1, norm_x2, norm_y2, img_height, img_width  = bf.calculate_bbox_and_draw2(input_image_path)

            # 重心、bboxをcsvに保存
            writer.writerow([i, x1, y1, x2, y2, norm_x1, norm_y1, norm_x2, norm_y2, img_height, img_width])

if __name__ == '__main__':
    # dataset_dir = "C:/workspace/MasterResearch/blender_dataset/20240812_2_512x512"
    dataset_dir = 'C:/workspace/Github/master-research/dataset/002'
    main2(dataset_dir)
