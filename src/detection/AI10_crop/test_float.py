# # 学習済みのpythonモデルを使って、画像の推論を行う
# # 量子化前のモデルを使う

# import tensorflow as tf
# from keras.layers import (
#     Dense,
#     Input,
#     GlobalAveragePooling2D,
#     Dropout,
#     BatchNormalization,
#     UpSampling2D,
#     Conv2D,
#     MaxPooling2D,
#     AveragePooling2D,
#     Flatten,
#     Concatenate,
# )
# from tensorflow.keras.optimizers import Adam
# from keras.applications.vgg16 import VGG16, preprocess_input
# from keras.preprocessing.image import ImageDataGenerator
# from keras.callbacks import CSVLogger
# from keras.models import Model
# from tensorflow.keras.applications import MobileNet, MobileNetV2
# from tensorflow import keras
# from keras.callbacks import ModelCheckpoint


# # import tensorflow_model_optimization as tfmot
# from keras import backend as K
# from tqdm import tqdm
# import os
# import sys
# import numpy as np
# import h5py
# import matplotlib.pyplot as plt
# import cv2
# from module.const import *
# from module import func, loss
# import random
# from keras.models import load_model
# import keras.backend as K

# from module import func, loss
# from module.const import *

# # dataset path
# # input
# test_img_dir = "../../../dataset/20240624_dataset/test/img"
# test_label_dir = "../../../dataset/20240624_dataset/test/label"

# # output 3枚の画像を並べて保存
# test_result_dir = "../../../test_result/20240710"

# # model path
# model_path = "mater_model/best_model.h5"

# # load model
# model = load_model(model_path, custom_objects={'cross_loss': loss.cross_loss, 'IoU': loss.IoU})



# # test model
# for i, (img, label) in enumerate(zip(tqdm(os.listdir(test_img_dir)), os.listdir(test_label_dir))):
#     img = os.path.join(test_img_dir, img)
#     img = cv2.imread(img)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, INPUT_SIZE)
#     label = os.path.join(test_label_dir, label)
#     label = cv2.imread(label)
#     # label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
#     # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img = np.array([img]) / 255
#     pred = model.predict(img)[0]
#     # pred = np.argmax(pred, axis=2) #
#     pred = pred[:, :, 1]
#     # pred = np.array(pred * 255, dtype=np.uint8)
#     # cv2.imwrite(os.path.join(test_result_dir, "result.png"), pred)
#     plt.subplot(1, 3, 1)
#     # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)　# これを入れると色がおかしくなる？？
#     plt.imshow(img[0], cmap="gray")
#     plt.subplot(1, 3, 2)
#     plt.imshow(label, cmap="gray")
#     plt.subplot(1, 3, 3)
#     plt.imshow(pred, vmin=0, vmax=1)
#     # plt.show()
#     plt.savefig(os.path.join(test_result_dir, f"{i}.png"))
#     plt.close()

import numpy as np
import csv
import os
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
from module import func, loss
from module.const import *
import math

def calculate_iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

# dataset path
test_img_dir = "../../../dataset/20240624_dataset/valid/input"
test_label_dir = "../../../dataset/20240624_dataset/valid/bin_label"

# output directory
test_result_dir = "../../../valid_result/20240710_3"
if not os.path.exists(test_result_dir):
    os.makedirs(test_result_dir)

# model path
model_path = "master_model/best_model.h5"

# load model
model = load_model(model_path, custom_objects={'cross_loss': loss.cross_loss, 'IoU': loss.IoU})

# Initialize list to store IoU results
iou_results = []

# test model
for i, (img_file, label_file) in enumerate(zip(tqdm(os.listdir(test_img_dir)), os.listdir(test_label_dir))):
    img_path = os.path.join(test_img_dir, img_file)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, INPUT_SIZE)
    
    label_path = os.path.join(test_label_dir, label_file)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    label = cv2.resize(label, LABEL_SIZE)
    label = label / 255  # Normalize to 0-1
    
    img_normalized = np.array([img]) / 255
    pred = model.predict(img_normalized)[0]
    pred = pred[:, :, 1]
    
    # Calculate IoU
    iou = calculate_iou(label > 0.5, pred > 0.5)
    if math.isnan(iou) == True:
        iou = 0.0
    iou_results.append((img_file, iou))
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Input Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(label, cmap="gray")
    plt.title('Ground Truth')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(pred, cmap="gray", vmin=0, vmax=1)
    plt.title(f'Prediction (IoU: {iou:.4f})')
    plt.axis('off')
    
    plt.suptitle(f'Image {i+1}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(test_result_dir, f"{i}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # print(f"Image {i+1} - IoU: {iou:.4f}")

# Save IoU results to CSV
csv_path = os.path.join(test_result_dir, 'iou_results.csv')
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Image', 'IoU'])  # Header
    writer.writerows(iou_results)

print(f"IoU results saved to {csv_path}")

# Calculate and print average IoU
average_iou = np.mean([iou for _, iou in iou_results])
print(f"Average IoU: {average_iou:.4f}")