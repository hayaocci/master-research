import cv2
import os
from tqdm import tqdm

def resize_images(input_dir, output_dir, size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(input_dir):
        img = cv2.imread(os.path.join(input_dir, filename))
        img = cv2.resize(img, size)
        cv2.imwrite(os.path.join(output_dir, filename), img)

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

def main():
    dataset_dir = 'dataset/001'
    dataset_train_dir = os.path.join(dataset_dir, 'train')
    dataset_valid_dir = os.path.join(dataset_dir, 'valid')
    dataset_train_input = os.path.join(dataset_train_dir, 'input')
    dataset_train_output = os.path.join(dataset_train_dir, 'output')
    dataset_valid_input = os.path.join(dataset_valid_dir, 'input')
    dataset_valid_output = os.path.join(dataset_valid_dir, 'output')

    # 出力
    dataset_resized_dir = 'dataset/003'
    dataset_resized_train_dir = os.path.join(dataset_resized_dir, 'train')
    dataset_resized_valid_dir = os.path.join(dataset_resized_dir, 'valid')
    dataset_resized_train_input = os.path.join(dataset_resized_train_dir, 'input')
    dataset_resized_train_output = os.path.join(dataset_resized_train_dir, 'output')
    dataset_resized_valid_input = os.path.join(dataset_resized_valid_dir, 'input')
    dataset_resized_valid_output = os.path.join(dataset_resized_valid_dir, 'output')

    # ディレクトリの作成
    if not os.path.exists(dataset_resized_dir):
        os.makedirs(dataset_resized_dir)
    if not os.path.exists(dataset_resized_train_dir):
        os.makedirs(dataset_resized_train_dir)
    if not os.path.exists(dataset_resized_valid_dir):
        os.makedirs(dataset_resized_valid_dir)
    if not os.path.exists(dataset_resized_train_input):
        os.makedirs(dataset_resized_train_input)
    if not os.path.exists(dataset_resized_train_output):
        os.makedirs(dataset_resized_train_output)
    if not os.path.exists(dataset_resized_valid_input):
        os.makedirs(dataset_resized_valid_input)
    if not os.path.exists(dataset_resized_valid_output):
        os.makedirs(dataset_resized_valid_output)

    # 画像のリサイズ
    size = (96, 96)

   # Trainデータのリサイズ
    file_count = get_file_num(dataset_train_input)
    print(f'file_count: {file_count}')
    for count in tqdm(range(file_count), desc='Resizing train input images'):
        input_file = os.path.join(dataset_train_input, f'{count}.png')
        output_file = os.path.join(dataset_resized_train_input, f'{count}.png')
        img = cv2.imread(input_file)
        img = cv2.resize(img, size)
        cv2.imwrite(output_file, img)

    file_count = get_file_num(dataset_train_output)
    print(f'file_count: {file_count}')
    for count in tqdm(range(file_count), desc='Resizing train output images'):
        input_file = os.path.join(dataset_train_output, f'{count}.png')
        output_file = os.path.join(dataset_resized_train_output, f'{count}.png')
        img = cv2.imread(input_file)
        img = cv2.resize(img, size)
        cv2.imwrite(output_file, img)

    # Validデータのリサイズ
    file_count = get_file_num(dataset_valid_input)
    print(f'file_count: {file_count}')
    for count in tqdm(range(file_count), desc='Resizing valid input images'):
        input_file = os.path.join(dataset_valid_input, f'{count}.png')
        output_file = os.path.join(dataset_resized_valid_input, f'{count}.png')
        img = cv2.imread(input_file)
        img = cv2.resize(img, size)
        cv2.imwrite(output_file, img)

    file_count = get_file_num(dataset_valid_output)
    print(f'file_count: {file_count}')
    for count in tqdm(range(file_count), desc='Resizing valid output images'):
        input_file = os.path.join(dataset_valid_output, f'{count}.png')
        output_file = os.path.join(dataset_resized_valid_output, f'{count}.png')
        img = cv2.imread(input_file)
        img = cv2.resize(img, size)
        cv2.imwrite(output_file, img)

if __name__ == '__main__':
    main()






