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
LEARNING_RATE = 0.001

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
train_norm_cx_list = utils.make_label_array(header, train_rows, 'norm_cx')
train_norm_cy_list = utils.make_label_array(header, train_rows, 'norm_cy')
train_norm_w_list = utils.make_label_array(header, train_rows, 'norm_w')
train_norm_h_list = utils.make_label_array(header, train_rows, 'norm_h')

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
val_norm_cx_list = utils.make_label_array(header, val_rows, 'norm_cx')
val_norm_cy_list = utils.make_label_array(header, val_rows, 'norm_cy')
val_norm_w_list = utils.make_label_array(header, val_rows, 'norm_w')
val_norm_h_list = utils.make_label_array(header, val_rows, 'norm_h')

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

for i in range(len(train_list)):
    for j in range(len(train_list[i])):
        if train_list[i][j] < 0 or train_list[i][j] > 1:
            print('Error: train_list is not normalized')
            exit()

for i in range(len(valid_list)):
    for j in range(len(valid_list[i])):
        if valid_list[i][j] < 0 or valid_list[i][j] > 1:
            print('Error: valid_list is not normalized')
            exit()

print('=====================================Label array and image number are matched=====================================')

# %%


# def create_model(input_shape):
#     # モバイルネットV2のベースモデルを読み込み
#     base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    
#     # ベースモデルのパラメータを凍結
#     base_model.trainable = False

#     # 入力層
#     inputs = Input(shape=input_shape)

#     # ベースモデルに入力を渡す
#     x = base_model(inputs, training=False)
    
#     # グローバル平均プーリング
#     x = GlobalAveragePooling2D()(x)
    
#     # 全結合層
#     x = Dense(1024, activation='relu')(x)
#     x = Dense(512, activation='relu')(x)
    
#     # 出力層: 4つのノードでcy,cy,w,h座標を予測
#     outputs = Dense(4, activation='sigmoid')(x)

#     # モデルの作成
#     model = Model(inputs, outputs)
    
#     return model

# モデルのインスタンス化
# model_crop = create_model(INPUT_SHAPE)

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
output = keras.layers.Dense(4, activation='linear')(x)  # 2つのニューロンで (x, y) 座標を出力

# 新しいモデルを構築
model_crop = keras.Model(inputs=intermediate_layer_model.input, outputs=output)

model_crop.trainable = True

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

def iou(y_true, y_pred):
    def calc_iou(box1, box2):
        # IoUの計算
        x1, y1, w1, h1 = box1[0], box1[1], box1[2], box1[3]
        x2, y2, w2, h2 = box2[0], box2[1], box2[2], box2[3]
        
        x1_min, y1_min, x1_max, y1_max = x1 - w1/2, y1 - h1/2, x1 + w1/2, y1 + h1/2
        x2_min, y2_min, x2_max, y2_max = x2 - w2/2, y2 - h2/2, x2 + w2/2, y2 + h2/2
        
        intersection_x_min = tf.maximum(x1_min, x2_min)
        intersection_y_min = tf.maximum(y1_min, y2_min)
        intersection_x_max = tf.minimum(x1_max, x2_max)
        intersection_y_max = tf.minimum(y1_max, y2_max)
        
        intersection_area = tf.maximum(0., intersection_x_max - intersection_x_min) * \
                            tf.maximum(0., intersection_y_max - intersection_y_min)
        
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - intersection_area
        
        return intersection_area / (union_area + keras.backend.epsilon())
    
    iou_values = tf.map_fn(lambda x: calc_iou(x[0], x[1]), (y_true, y_pred), dtype=tf.float32)
    return tf.reduce_mean(iou_values)

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

# モデルのコンパイル
model_crop.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse', metrics=iou)

# CVSLoggerを使用してログを保存する
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file_name = 'crop_model_linear_{}_{}_{}_{}.csv'.format(EPOCHS, BATCH_SIZE, LEARNING_RATE, IMG_SIZE[0])

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
                filepath='trained_crop_model/trained_crop_best_model_linear_{}_{}_{}_{}.h5'.format(BATCH_SIZE, EPOCHS, LEARNING_RATE, IMG_SIZE[0]),
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
