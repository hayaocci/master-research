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
if img_size[0] != 224 or img_size[1] != 224:
    print('Error: image size is not 224x224')

    for image in tqdm(os.listdir(train_image_dir)):
        image_path = os.path.join(train_image_dir, image)
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        train_image.append(img)
else:
    for image in tqdm(os.listdir(train_image_dir)):
        image_path = os.path.join(train_image_dir, image)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        train_image.append(img)

train_image = np.array(train_image) / 255.0

print('valid image making...')
valid_image = []

val_sample_img = cv2.imread(os.path.join(val_image_dir, '0.png'))
img_size = val_sample_img.shape[:2]
if img_size[0] != 224 or img_size[1] != 224:
    print('Error: image size is not 224x224')

    for image in tqdm(os.listdir(val_image_dir)):
        image_path = os.path.join(val_image_dir, image)
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = img / 255.0
        # valid_image = valid_image / 255.0
        valid_image.append(img)
else:
    for image in tqdm(os.listdir(val_image_dir)):
        image_path = os.path.join(val_image_dir, image)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = img / 255.0
        # valid_image = np.array(valid_image) / 255.0
        # valid_image = valid_image / 255.0
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

print("train_list: ", train_list[0], train_list[1])
    
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
モデルの定義
"""
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
output = keras.layers.Dense(2, activation='sigmoid')(x)  # 2つのニューロンで (x, y) 座標を出力

# 新しいモデルを構築
model_with_coordinates = keras.Model(inputs=intermediate_layer_model.input, outputs=output)

# 新しいモデルの構造を確認
model_with_coordinates.summary()
total_params = model_with_coordinates.count_params()
print(f'Total params: {total_params}')

# モデルの保存
model_with_coordinates.save('custom_model/coordinates_model.h5')
print('=====================================Model is defined=====================================')

# %%
"""
学習を行う
"""

loss = keras.losses.MeanSquaredError()

# BATCH_SIZE = 32
# EPOCHS = 50
# LEARNING_RATE = 0.0005

# model_with_coordinates.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss=loss)

# model_with_coordinates.fit(
#     train_image, # 画像データ
#     train_list, # ラベルデータ
#     batch_size=BATCH_SIZE,
#     epochs=EPOCHS,
#     validation_data=(valid_image, valid_list),
#     callbacks=[
#         ModelCheckpoint(
#             filepath='trained_custom_model/trained_coordinates_best_model_{}_{}_{}.h5'.format(BATCH_SIZE, EPOCHS, LEARNING_RATE),
#             save_best_only=True,
#             monitor='val_loss',
#             mode='min'
#         ),
#         CSVLogger('log/coordinates_model_{}_{}_{}.csv'.format(BATCH_SIZE, EPOCHS, LEARNING_RATE))
#     ],
# )

# モデルの損失関数
loss = keras.losses.MeanSquaredError()

# ハイパーパラメータ
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 0.0005

# モデルのコンパイル
model_with_coordinates.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss=loss)

# CSVLoggerを使用してログを保存する
log_filename = 'log/coordinates_model_log.csv'

with open(log_filename, 'w', encoding='utf-8', newline='') as csvfile:
    csv_logger = CSVLogger(log_filename)

    # モデルの学習
    model_with_coordinates.fit(
        train_image,  # 画像データ
        train_list,   # ラベルデータ
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(valid_image, valid_list),
        callbacks=[
            ModelCheckpoint(
                filepath='trained_custom_model/trained_coordinates_best_model_{}_{}_{}.h5'.format(BATCH_SIZE, EPOCHS, LEARNING_RATE),
                save_best_only=True,
                monitor='val_loss',
                mode='min'
            ),
            csv_logger
        ],
    )

# モデルの学習が終わった後に呼び出す
tf.keras.backend.clear_session()


# BATCH_SIZE = 32
# EPOCHS = 100
# LEARNING_RATE = 0.002

# model_with_coordinates.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mean_squared_error')

# model_with_coordinates.fit(
#     train_image, # 画像データ
#     train_list, # ラベルデータ
#     batch_size=BATCH_SIZE,
#     epochs=EPOCHS,
#     validation_data=(valid_image, valid_list),
#     callbacks=[
#         ModelCheckpoint(
#             filepath='trained_custom_model/trained_coordinates_best_model_{}_{}_{}.h5'.format(BATCH_SIZE, EPOCHS, LEARNING_RATE),
#             save_best_only=True,
#             monitor='val_loss',
#             mode='min'
#         ),
#         CSVLogger('log/coordinates_model_{}_{}_{}.csv'.format(BATCH_SIZE, EPOCHS, LEARNING_RATE))
#     ],
# )

# # モデルの学習が終わった後に呼び出す
# tf.keras.backend.clear_session()

# BATCH_SIZE = 32
# EPOCHS = 100
# LEARNING_RATE = 0.003

# model_with_coordinates.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mean_squared_error')

# model_with_coordinates.fit(
#     train_image, # 画像データ
#     train_list, # ラベルデータ
#     batch_size=BATCH_SIZE,
#     epochs=EPOCHS,
#     validation_data=(valid_image, valid_list),
#     callbacks=[
#         ModelCheckpoint(
#             filepath='trained_custom_model/trained_coordinates_best_model_{}_{}_{}.h5'.format(BATCH_SIZE, EPOCHS, LEARNING_RATE),
#             save_best_only=True,
#             monitor='val_loss',
#             mode='min'
#         ),
#         CSVLogger('log/coordinates_model_{}_{}_{}.csv'.format(BATCH_SIZE, EPOCHS, LEARNING_RATE))
#     ],
# )

# # モデルの学習が終わった後に呼び出す
# tf.keras.backend.clear_session()

# BATCH_SIZE = 32
# EPOCHS = 100
# LEARNING_RATE = 0.004

# model_with_coordinates.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mean_squared_error')

# model_with_coordinates.fit(
#     train_image, # 画像データ
#     train_list, # ラベルデータ
#     batch_size=BATCH_SIZE,
#     epochs=EPOCHS,
#     validation_data=(valid_image, valid_list),
#     callbacks=[
#         ModelCheckpoint(
#             filepath='trained_custom_model/trained_coordinates_best_model_{}_{}_{}.h5'.format(BATCH_SIZE, EPOCHS, LEARNING_RATE),
#             save_best_only=True,
#             monitor='val_loss',
#             mode='min'
#         ),
#         CSVLogger('log/coordinates_model_{}_{}_{}.csv'.format(BATCH_SIZE, EPOCHS, LEARNING_RATE))
#     ],
# )

# # モデルの学習が終わった後に呼び出す
# tf.keras.backend.clear_session()

# BATCH_SIZE = 32
# EPOCHS = 100
# LEARNING_RATE = 0.005
# model_with_coordinates.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mean_squared_error')

# model_with_coordinates.fit(
#     train_image, # 画像データ
#     train_list, # ラベルデータ
#     batch_size=BATCH_SIZE,
#     epochs=EPOCHS,
#     validation_data=(valid_image, valid_list),
#     callbacks=[
#         ModelCheckpoint(
#             filepath='trained_custom_model/trained_coordinates_best_model_{}_{}_{}.h5'.format(BATCH_SIZE, EPOCHS, LEARNING_RATE),
#             save_best_only=True,
#             monitor='val_loss',
#             mode='min'
#         ),
#         CSVLogger('log/coordinates_model_{}_{}_{}.csv'.format(BATCH_SIZE, EPOCHS, LEARNING_RATE))
#     ],
# )
# # モデルの学習が終わった後に呼び出す
# tf.keras.backend.clear_session()

# BATCH_SIZE = 32
# EPOCHS = 100
# LEARNING_RATE = 0.008
# model_with_coordinates.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mean_squared_error')

# model_with_coordinates.fit(
#     train_image, # 画像データ
#     train_list, # ラベルデータ
#     batch_size=BATCH_SIZE,
#     epochs=EPOCHS,
#     validation_data=(valid_image, valid_list),
#     callbacks=[
#         ModelCheckpoint(
#             filepath='trained_custom_model/trained_coordinates_best_model_{}_{}_{}.h5'.format(BATCH_SIZE, EPOCHS, LEARNING_RATE),
#             save_best_only=True,
#             monitor='val_loss',
#             mode='min'
#         ),
#         CSVLogger('log/coordinates_model_{}_{}_{}.csv'.format(BATCH_SIZE, EPOCHS, LEARNING_RATE))
#     ],
# )

# # モデルの学習が終わった後に呼び出す
# tf.keras.backend.clear_session()

# BATCH_SIZE = 32
# EPOCHS = 100
# LEARNING_RATE = 0.01
# model_with_coordinates.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mean_squared_error')

# model_with_coordinates.fit(
#     train_image, # 画像データ
#     train_list, # ラベルデータ
#     batch_size=BATCH_SIZE,
#     epochs=EPOCHS,
#     validation_data=(valid_image, valid_list),
#     callbacks=[
#         ModelCheckpoint(
#             filepath='trained_custom_model/trained_coordinates_best_model_{}_{}_{}.h5'.format(BATCH_SIZE, EPOCHS, LEARNING_RATE),
#             save_best_only=True,
#             monitor='val_loss',
#             mode='min'
#         ),
#         CSVLogger('log/coordinates_model_{}_{}_{}.csv'.format(BATCH_SIZE, EPOCHS, LEARNING_RATE))
#     ],
# )

# %%
model_with_coordinates.save('trained_custom_model/trained_coordinates_model_last.h5')










# def load_data(image_dir, csv_path):
#     images = []
#     coordinates = []
    
#     df = pd.read_csv(csv_path)
    
#     for index, row in df.iterrows():
#         img_path = os.path.join(image_dir, row['image_name'])
#         img = cv2.imread(img_path)
#         img = cv2.resize(img, INPUT_SIZE)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
#         images.append(img)
#         coordinates.append([row['x'], row['y']])
    
#     return np.array(images), np.array(coordinates)

# # データの読み込み
# X, y = load_data('path/to/image/directory', 'path/to/coordinates.csv')

# # データの分割
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # モデルの定義
# # def create_model():
# #     input_layer = Input(shape=(INPUT_SIZE[0], INPUT_SIZE[1], INPUT_CHANNEL))
    
# #     x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
# #     x = BatchNormalization()(x)
# #     x = MaxPooling2D((2, 2))(x)
    
# #     x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
# #     x = BatchNormalization()(x)
# #     x = MaxPooling2D((2, 2))(x)
    
# #     x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
# #     x = BatchNormalization()(x)
# #     x = MaxPooling2D((2, 2))(x)
    
# #     x = GlobalAveragePooling2D()(x)
# #     x = Dense(128, activation='relu')(x)
# #     x = Dropout(0.5)(x)
    
# #     output_layer = Dense(2, activation='linear')(x)  # x, y 座標を出力
    
# #     model = Model(inputs=input_layer, outputs=output_layer)
# #     return model

# """
# モデルの定義
# """
# def create_model(base_model):
#     input_layer = base_model.input
    
#     x = base_model.get_layer('block5_pool').output
#     x = GlobalAveragePooling2D()(x)
#     x = Dense(128, activation='relu')(x)
#     x = Dropout(0.5)(x)
    
#     output_layer = Dense(2, activation='linear')(x)  # x, y 座標を出力
    
#     model = Model(inputs=input_layer, outputs=output_layer)
#     return model

# """
# 学習を行う
# """


# # モデルのコンパイルと訓練
# model = create_model()
# model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# history = model.fit(
#     X_train, y_train,
#     batch_size=32,
#     epochs=50,
#     validation_split=0.2,
#     callbacks=[
#         keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss'),
#         keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
#     ]
# )

# # モデルの評価
# test_loss, test_mae = model.evaluate(X_test, y_test)
# print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

# # モデルの保存
# model.save('centroid_model.h5')

# # TFLiteモデルへの変換
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()

# with open('centroid_model.tflite', 'wb') as f:
#     f.write(tflite_model)
# %%
