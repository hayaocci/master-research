# %% 
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Input, GlobalAveragePooling2D, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from keras.models import Model
# from keras.optimizers import Adam
import numpy as np
import os
import cv2
import csv
from tqdm import tqdm

import tensorflow as tf
from keras.layers import (
    Dense,
    Input,
    GlobalAveragePooling2D,
    Dropout,
    BatchNormalization,
    UpSampling2D,
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    Flatten,
    Concatenate,
)
from tensorflow.keras.optimizers import Adam
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger
from keras.models import Model
from tensorflow.keras.applications import MobileNet, MobileNetV2
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, CSVLogger

import module.utils as utils




# データセットパス
# DATASET_MAIN_DIR = 'C:/workspace/MasterResearch/blender_dataset'
REPOSITORY_DIR = 'C:/workspace/Github/master-research'
DATASET_MAIN_DIR = os.path.join(REPOSITORY_DIR, 'dataset')
dataset_dir = os.path.join(DATASET_MAIN_DIR, '001') # 場合によって書き換える

"""
以下編集不要
"""
# 学習用ディレクトリ
train_dir = os.path.join(dataset_dir, 'train')
train_image_dir = os.path.join(train_dir, 'input')
train_csv_path = os.path.join(dataset_dir, 'train.csv')

# 検証用ディレクトリ
val_dir = os.path.join(dataset_dir, 'valid')
val_image_dir = os.path.join(val_dir, 'input')
val_csv_path = os.path.join(dataset_dir, 'valid.csv')

print('train image making...')
# train_image = []

# for image in tqdm(os.listdir(train_image_dir)):
#     image_path = os.path.join(train_image_dir, image)
#     img = cv2.imread(image_path)
#     img = cv2.resize(img, (224, 224))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     # img = img / 255.0
#     train_image.append(img)

# train_image = np.array(train_image) / 255.0

print('valid image making...')
valid_image = []

for image in tqdm(os.listdir(val_image_dir)):
    image_path = os.path.join(val_image_dir, image)
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = img / 255.0
    valid_image.append(img)

valid_image = np.array(valid_image) / 255.0

# %%
# データセットCSVの読み込み
header, train_rows = utils.read_csv(train_csv_path)
header, val_rows = utils.read_csv(val_csv_path)

# ラベルの作成
train_gx_list = utils.make_label_array(header, train_rows, 'gx')
train_gy_list = utils.make_label_array(header, train_rows, 'gy')
train_list = []
for i in range(len(train_gx_list)):
    train_list.append([float(train_gx_list[i]) / 512.0, float(train_gy_list[i]) / 512.0])

# ラベルの型チェック
if (type(train_gx_list) != list) or (type(train_gy_list) != list):
    print('Error: label array is not list')
    exit()

# ラベルの数と画像の数が一致しているか確認
if len(train_gx_list) != utils.get_file_num(train_image_dir):
    print('Error: label array and image number are not matched')
    exit()
if len(train_gy_list) != utils.get_file_num(train_image_dir):
    print('Error: label array and image number are not matched')
    exit()

train_list = np.array(train_list)

# ラベルの作成
val_gx_list = utils.make_label_array(header, val_rows, 'gx')
val_gy_list = utils.make_label_array(header, val_rows, 'gy')
valid_list = []
for i in range(len(val_gx_list)):
    valid_list.append([float(val_gx_list[i]) / 512.0, float(val_gy_list[i]) / 512.0])
    
# ラベルの型チェック
if (type(val_gx_list) != list) or (type(val_gy_list) != list):
    print('Error: label array is not list')
    exit()

# ラベルの数と画像の数が一致しているか確認
if len(val_gx_list) != utils.get_file_num(val_image_dir):
    print('Error: label array and image number are not matched')
    exit()
if len(val_gy_list) != utils.get_file_num(val_image_dir):
    print('Error: label array and image number are not matched')
    exit()

valid_list = np.array(valid_list) /255.0

print('=====================================Label array and image number are matched=====================================')

# %%
"""
学習済みモデルの読み込み
"""

# モデルの読み込み
train_model = keras.models.load_model(REPOSITORY_DIR + '/src/detection/AI10_crop/trained_custom_model/trained_coordinates_best_model_32_100.h5')

# モデルのサマリー
train_model.summary()

# モデルで予測
predictions = train_model.predict(valid_image)

predictions = predictions * 512.0

output_csv_path = os.path.join('validation_predictions_32_50_0.0005.csv')

with open(output_csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['gx', 'gy'])  # ヘッダーを書き込む
    for prediction in predictions:
        writer.writerow(prediction)