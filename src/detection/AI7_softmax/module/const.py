import os

INPUT_SIZE = (96, 96)
NUM_AREA = 12  # 12の約数にすること
LABEL_SIZE = (NUM_AREA, NUM_AREA)
INPUT_CHANNEL = 3
ALPHA = 0.75  # 1はダメ0.5は行ける
SPLIT_NUM = 1

RAW_DATA_DIR = "rawdata"  # 生データ
DATA_DIR = "../AI6_softmax/master_data"  # データセット。生データを変換したもの

# dataset path
dataset_main_dir = "../../../dataset"
dataset_name = "20240624_dataset" # 主にここを変更する
dataset_dir = os.path.join(dataset_main_dir, dataset_name)
train_dir = os.path.join(dataset_dir, "train")
valid_dir = os.path.join(dataset_dir, "valid")



# 保存するモデルのパス
MODEL_DIR = "master_model/"
SPRESENSE_MODEL_DIR = "../../Spresense/detect_people/"
FULL_MODEL_PATH = os.path.join(MODEL_DIR, "model.h5")
TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, "model.tflite")
TFLITE_QUANT_MODEL_PATH = os.path.join(MODEL_DIR, "model_quant.tflite")
HEADER_MODEL_PATH = os.path.join(MODEL_DIR, "spresense_model.h")
HEADER_QUANT_MODEL_PATH = os.path.join(MODEL_DIR, "spresense_model_quant2.h")

SPRESENSE_HEADER_QUANT_MODEL_PATH = os.path.join(
    SPRESENSE_MODEL_DIR, "spresense_model_quant.h"
)
SPRESENSE_HEADER_MODEL_PATH = os.path.join(SPRESENSE_MODEL_DIR, "spresense_model.h")


VALID_DIR = "../../../valid_result/20240710"
