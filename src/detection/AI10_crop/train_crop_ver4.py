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
dataset_dir = os.path.join(DATASET_MAIN_DIR, '002') # 場合によって書き換える

IMG_SIZE = [224, 224]
BASE_IMG_SIZE = 512.0
INPUT_SHAPE = (IMG_SIZE[0], IMG_SIZE[1], 3)

BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 0.0005

"""
以下編集不要
"""
# 学習用ディレクトリ
train_dir = os.path.join(dataset_dir, 'train')
train_image_dir = os.path.join(train_dir, 'input')
train_csv_path = os.path.join(dataset_dir, 'train_2.csv')

# 検証用ディレクトリ
val_dir = os.path.join(dataset_dir, 'valid')
val_image_dir = os.path.join(val_dir, 'input')
val_csv_path = os.path.join(dataset_dir, 'valid_2.csv')

print('train image making...')
train_image = []

train_sample_img = cv2.imread(os.path.join(train_image_dir, '0.png'))
img_size = train_sample_img.shape[:2]
if img_size != (IMG_SIZE[0], IMG_SIZE[1]):
    print('Error: image size is not matched')

    for image in tqdm(os.listdir(train_image_dir)):
        image_path = os.path.join(train_image_dir, image)
        img = cv2.imread(image_path)
        img = cv2.resize(img, (IMG_SIZE[0], IMG_SIZE[1]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        train_image.append(img)
else:
    for image in tqdm(os.listdir(train_image_dir)):
        image_path = os.path.join(train_image_dir, image)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        train_image.append(img)

train_image = np.array(train_image) / 255.0

print('==========train image making done==========')

print('valid image making...')
valid_image = []

valid_sample_img = cv2.imread(os.path.join(val_image_dir, '0.png'))
img_size = valid_sample_img.shape[:2]
if img_size != (IMG_SIZE[0], IMG_SIZE[1]):
    print('Error: image size is not matched')

    for image in tqdm(os.listdir(val_image_dir)):
        image_path = os.path.join(val_image_dir, image)
        img = cv2.imread(image_path)
        img = cv2.resize(img, (IMG_SIZE[0], IMG_SIZE[1]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        valid_image.append(img)
else:
    for image in tqdm(os.listdir(val_image_dir)):
        image_path = os.path.join(val_image_dir, image)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        valid_image.append(img)

valid_image = np.array(valid_image) / 255.0

print('==========valid image making done==========')

# %%
# データセットCSVの読み込み
header, train_rows = utils.read_csv(train_csv_path)
header, val_rows = utils.read_csv(val_csv_path)

# ラベルの作成
# train_norm_cx_list = utils.make_label_array(header, train_rows, 'norm_x1')
# train_norm_cy_list = utils.make_label_array(header, train_rows, 'norm_y1')
# train_norm_w_list = utils.make_label_array(header, train_rows, 'norm_x2')
# train_norm_h_list = utils.make_label_array(header, train_rows, 'norm_y2')

train_norm_cx_list = utils.make_label_array(header, train_rows, 'x1')
train_norm_cy_list = utils.make_label_array(header, train_rows, 'y1')
train_norm_w_list = utils.make_label_array(header, train_rows, 'x2')
train_norm_h_list = utils.make_label_array(header, train_rows, 'y2')

train_list = []
for i in range(len(train_norm_cx_list)):
    # train_list.append([float(train_norm_cx_list[i])/BASE_IMG_SIZE, float(train_norm_cy_list[i])/BASE_IMG_SIZE, float(train_norm_w_list[i])/BASE_IMG_SIZE, float(train_norm_h_list[i])/BASE_IMG_SIZE])
    train_list.append([float(train_norm_cx_list[i]), float(train_norm_cy_list[i]), float(train_norm_w_list[i]), float(train_norm_h_list[i])])

# ラベルの型チェック
if (type(train_norm_cx_list) != list) or (type(train_norm_cy_list) != list) or (type(train_norm_w_list) != list) or (type(train_norm_h_list) != list):
    print('Error: label array is not list')
    exit()

# ラベルの数と画像の数が一致しているか確認
if len(train_norm_cx_list) != utils.get_file_num(train_image_dir):
    print('Error: label array and image number are not matched')
    exit()
if len(train_norm_cy_list) != utils.get_file_num(train_image_dir):
    print('Error: label array and image number are not matched')
    exit()
if len(train_norm_w_list) != utils.get_file_num(train_image_dir):
    print('Error: label array and image number are not matched')
    exit()
if len(train_norm_h_list) != utils.get_file_num(train_image_dir):
    print('Error: label array and image number are not matched')
    exit()


train_list = np.array(train_list)

# ラベルの作成
# val_norm_cx_list = utils.make_label_array(header, val_rows, 'norm_x1')
# val_norm_cy_list = utils.make_label_array(header, val_rows, 'norm_y1')
# val_norm_w_list = utils.make_label_array(header, val_rows, 'norm_x2')
# val_norm_h_list = utils.make_label_array(header, val_rows, 'norm_y2')

val_norm_cx_list = utils.make_label_array(header, val_rows, 'x1')
val_norm_cy_list = utils.make_label_array(header, val_rows, 'y1')
val_norm_w_list = utils.make_label_array(header, val_rows, 'x2')
val_norm_h_list = utils.make_label_array(header, val_rows, 'y2')

valid_list = []
for i in range(len(val_norm_cx_list)):
    # valid_list.append([float(val_norm_cx_list[i])/BASE_IMG_SIZE, float(val_norm_cy_list[i])/BASE_IMG_SIZE, float(val_norm_w_list[i])/BASE_IMG_SIZE, float(val_norm_h_list[i])/BASE_IMG_SIZE])
    valid_list.append([float(val_norm_cx_list[i]), float(val_norm_cy_list[i]), float(val_norm_w_list[i]), float(val_norm_h_list[i])])

# ラベルの型チェック
if (type(val_norm_cx_list) != list) or (type(val_norm_cy_list) != list) or (type(val_norm_w_list) != list) or (type(val_norm_h_list) != list):
    print('Error: label array is not list')
    exit()

# ラベルの数と画像の数が一致しているか確認
if len(val_norm_cx_list) != utils.get_file_num(val_image_dir):
    print('Error: label array and image number are not matched')
    exit()
if len(val_norm_cy_list) != utils.get_file_num(val_image_dir):
    print('Error: label array and image number are not matched')
    exit()
if len(val_norm_w_list) != utils.get_file_num(val_image_dir):
    print('Error: label array and image number are not matched')
    exit()
if len(val_norm_h_list) != utils.get_file_num(val_image_dir):
    print('Error: label array and image number are not matched')
    exit()

valid_list = np.array(valid_list)

# train_list と valid_list の大きさが0から1の範囲に収まっているか確認
print('train_list sample:', train_list[0])
print('valid_list sample:', valid_list[0])

# for i in range(len(train_list)):
#     for j in range(len(train_list[i])):
#         if train_list[i][j] < 0 or train_list[i][j] > 1:
#             print('Error: train_list is not normalized')
#             exit()

# for i in range(len(valid_list)):
#     for j in range(len(valid_list[i])):
#         if valid_list[i][j] < 0 or valid_list[i][j] > 1:
#             print('Error: valid_list is not normalized')
#             exit()

print('=====================================Label array and image number are matched=====================================')

# %%


def create_model(input_shape, active_func: str):
    # モバイルネットV2のベースモデルを読み込み
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # ベースモデルのパラメータを凍結
    base_model.trainable = False

    # 入力層
    inputs = Input(shape=input_shape)

    # ベースモデルに入力を渡す
    x = base_model(inputs, training=False)
    
    # グローバル平均プーリング
    x = GlobalAveragePooling2D()(x)
    
    # 全結合層
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    
    # 出力層: 4つのノードでcy,cy,w,h座標を予測
    outputs = Dense(4, activation=active_func)(x)

    # モデルの作成
    model = Model(inputs, outputs)
    
    return model

# モデルのインスタンス化
# model_crop = create_model(INPUT_SHAPE)

# 活性化関数の定義
active_func = 'sigmoid'


# ベースモデルの読み込み
complessed_mobilenet = keras.models.load_model('model/base_model.h5')
# complessed_mobilenet = keras.models.load_model('model/base_model_96_cut.h5')
layer_name = 'block_6_expand_relu'  # 削除したい層の前の層の名前
intermediate_layer_model = keras.Model(inputs=complessed_mobilenet.input,
                                       outputs=complessed_mobilenet.get_layer(layer_name).output)
# complessed_mobilenet.summary()
# intermediate_layer_model.summary()

# グローバル平均プーリング層を追加
x = keras.layers.GlobalAveragePooling2D()(intermediate_layer_model.output)

# 全結合層を追加して (x, y) 座標を出力
output = keras.layers.Dense(4, activation=active_func)(x)  # 2つのニューロンで (x, y) 座標を出力

# 新しいモデルを構築
model_crop = keras.Model(inputs=intermediate_layer_model.input, outputs=output)


model_crop = create_model(INPUT_SHAPE, active_func)
# model_crop.trainable = True

model_crop.summary()
total_params = model_crop.count_params()
print(f'Total params: {total_params}')

# モデルの保存
model_crop.save('custom_model/crop_model.h5')
print('=====================================Model is defined=====================================')

# %%
"""
学習を行う
"""

# IOUの計算
def iou(y_val, y_pred):
    # 予測値と正解値の重なっている部分の面積
    x1 = tf.maximum(y_val[:, 0], y_pred[:, 0])
    y1 = tf.maximum(y_val[:, 1], y_pred[:, 1])
    x2 = tf.minimum(y_val[:, 2], y_pred[:, 2])
    y2 = tf.minimum(y_val[:, 3], y_pred[:, 3])
    
    # 重なっている部分の幅と高さ
    w = tf.maximum(0.0, x2 - x1)
    h = tf.maximum(0.0, y2 - y1)
    
    # 重なっている部分の面積
    intersection = w * h
    
    # 予測値と正解値の外接矩形の面積
    area_val = (y_val[:, 2] - y_val[:, 0]) * (y_val[:, 3] - y_val[:, 1])
    area_pred = (y_pred[:, 2] - y_pred[:, 0]) * (y_pred[:, 3] - y_pred[:, 1])
    
    # 重なっている部分の面積 / 予測値と正解値の外接矩形の面積
    union = area_val + area_pred - intersection
    iou = intersection / union
    
    return tf.reduce_mean(iou)

def mse(y_val, y_pred):
    # 1/n Σ(y1 - y2)^2
    loss = tf.reduce_mean(tf.square(y_val - y_pred))
    return loss

def binary_cross_entropy(y_val, y_pred):
    # int型だとエラーになりました
    y_val = tf.cast(y_val, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-6, 1-1e-6)  # log(0)log(1)回避用
    # - (p log(q) + (1-p) log(1-q))
    loss = - (y_val * tf.math.log(y_pred) + (1-y_val) * tf.math.log(1 - y_pred))
    return tf.reduce_mean(loss)

import tensorflow as tf

def iou_loss(y_true, y_pred):
    # 座標を取得
    x1_true, y1_true, x2_true, y2_true = y_true[:, 0], y_true[:, 1], y_true[:, 2], y_true[:, 3]
    x1_pred, y1_pred, x2_pred, y2_pred = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], y_pred[:, 3]

    # 交差領域の左上隅と右下隅を計算
    x1_intersection = tf.maximum(x1_true, x1_pred)
    y1_intersection = tf.maximum(y1_true, y1_pred)
    x2_intersection = tf.minimum(x2_true, x2_pred)
    y2_intersection = tf.minimum(y2_true, y2_pred)

    # 交差領域の幅と高さを計算
    intersection_width = tf.maximum(0.0, x2_intersection - x1_intersection)
    intersection_height = tf.maximum(0.0, y2_intersection - y1_intersection)

    # 交差領域の面積を計算
    intersection_area = intersection_width * intersection_height

    # 各ボックスの面積を計算
    true_area = (x2_true - x1_true) * (y2_true - y1_true)
    pred_area = (x2_pred - x1_pred) * (y2_pred - y1_pred)

    # 合計領域の面積を計算
    union_area = true_area + pred_area - intersection_area

    # IoUを計算
    iou = intersection_area / (union_area + tf.keras.backend.epsilon())

    # IoU Lossを計算 (1 - IoU)
    loss = 1 - iou

    # 平均損失
    return tf.reduce_mean(loss)

from module import loss

# モデルのコンパイル
model_crop.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss=iou_loss, metrics=iou)

# CVSLoggerを使用してログを保存する
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file_name = 'crop_model_v2_{}_{}_{}_{}_{}.csv'.format(active_func, EPOCHS, BATCH_SIZE, LEARNING_RATE, IMG_SIZE[0])

log_file_path = os.path.join(log_dir, log_file_name)

with open(log_file_path, 'w', encoding='utf-8', newline='') as csvfile:
    csv_logger = CSVLogger(log_file_path)

    # モデルの学習
    model_crop.fit(
        train_image,  # 画像データ
        train_list,   # ラベルデータ
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(valid_image, valid_list),
        callbacks=[
            ModelCheckpoint(
                filepath='trained_crop_model/trained_best_crop_model_v2_linear_{}_{}_{}_{}_{}.h5'.format(active_func, BATCH_SIZE, EPOCHS, LEARNING_RATE, IMG_SIZE[0]),
                verbose=1,
                save_best_only=True,
                save_weights_only=False,
                monitor='val_iou',
                mode='max',
                period=1,
            ),
            csv_logger
        ],
    )

# モデルの学習が終わった後に呼び出す
tf.keras.backend.clear_session()
print('=====================================Model is trained=====================================')
