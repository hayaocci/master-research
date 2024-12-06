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
from keras.callbacks import ModelCheckpoint


# import tensorflow_model_optimization as tfmot
from keras import backend as K
from tqdm import tqdm
import os
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
import cv2
from module.const import *
from module import func, loss
import random
from keras.models import load_model
import keras.backend as K

def pred_single_img(model_path, img_path, label=None):
    if model_path is None:
        print('Model path is None')
        return
    if img_path is None:
        print('Img path is None')
        return

    # Load model
    custom_model = load_model(model_path, custom_objects={'cross_loss': loss.cross_loss, 'IoU': loss.IoU})

    # Load image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, INPUT_SIZE)
    img = np.array([img]) / 255.0

    # Perform prediction
    pred = custom_model.predict(img)[0]

    # Visualization
    # plt.subplot(1, 2, 1)
    # plt.imshow(img[0])
    # plt.title('Input Image')
    # plt.subplot(1, 2, 2)
    # print(type(pred))
    # print(pred.shape)
    # plt.imshow(pred[:, :, 1], vmin=0, vmax=1)
    # plt.title('Prediction')
    # plt.savefig('result_single_img.png')
    # plt.close()

    pred_channel_1 = pred[:, :, 1]  # Second channel
    pred_channel_1 = (pred_channel_1 * 255).astype(np.uint8)
    pred_channel_1[pred_channel_1 > 0] = 255
    # cv2.imwrite('pred_channel_1.png', pred_channel_1)

    return pred_channel_1

val_dataset_dir = 'C:/workspace/Github/master-research/dataset/003/valid/input'
val_result_dir = 'valid_result/valid_20240908'
# custom_model_path = "master_model/trained_model.h5" # tensorflowのモデル
custom_model_path = "master_model/new_best_model_100_32_0.0005.h5"

if not os.path.exists(val_result_dir):
    os.makedirs(val_result_dir)

for i in range(2000):
    img_path = os.path.join(val_dataset_dir, f'{i}.png')
    pred = pred_single_img(custom_model_path, img_path)
    cv2.imwrite(os.path.join(val_result_dir, f'{i}.png'), pred)
    # print(f'{i}.png saved')

print('Done')
