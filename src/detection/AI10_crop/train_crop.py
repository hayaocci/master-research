import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Input, GlobalAveragePooling2D, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
import os
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

# 定数の定義
INPUT_SIZE = (224, 224)  # 入力画像サイズ
INPUT_CHANNEL = 3  # RGBの場合は3

def load_data(image_dir, csv_path):
    images = []
    coordinates = []
    
    df = pd.read_csv(csv_path)
    
    for index, row in df.iterrows():
        img_path = os.path.join(image_dir, row['image_name'])
        img = cv2.imread(img_path)
        img = cv2.resize(img, INPUT_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        images.append(img)
        coordinates.append([row['x'], row['y']])
    
    return np.array(images), np.array(coordinates)

# データの読み込み
X, y = load_data('path/to/image/directory', 'path/to/coordinates.csv')

# データの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# モデルの定義
def create_model():
    input_layer = Input(shape=(INPUT_SIZE[0], INPUT_SIZE[1], INPUT_CHANNEL))
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    output_layer = Dense(2, activation='linear')(x)  # x, y 座標を出力
    
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# モデルのコンパイルと訓練
model = create_model()
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=50,
    validation_split=0.2,
    callbacks=[
        keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss'),
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    ]
)

# モデルの評価
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

# モデルの保存
model.save('centroid_model.h5')

# TFLiteモデルへの変換
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('centroid_model.tflite', 'wb') as f:
    f.write(tflite_model)