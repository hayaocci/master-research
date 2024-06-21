"""
Blenderでデータセットを作成するスクリプト
地球とデブリのIoUで区別する
"""

import os
import numpy as np
import cv2
import random
import datetime as dt
import bpy
import bpycv
import csv
import time

# Constants
IMG_SIZE = (96, 96)
NUM_PIC = 100 # 100~ && x100
SPLIT_RATIO = 0.8
TRAIN_PIC = int(NUM_PIC * SPLIT_RATIO)
VALID_PIC = 11*200 #NUM_PIC - TRAIN_PIC

# colors
WHITE = [1, 1, 1, 1]
EXCOLOR = [0.445201, 0.201556, 0.0241577, 1]

current_datetiem = dt.datetime.now()
time_name = current_datetiem.strftime('%Y%m%d_%H%M%S')
dir_name = time_name + "_" + str(IMG_SIZE[0]) + "x" + str(IMG_SIZE[1])

# path
# SAVE_DIR = 'C:/workspace/senior_thesis/nnc001/dataset/'
SAVE_DIR = 'C:/workspace/dataset_all/'
DATASET_DIR = os.path.join(SAVE_DIR, dir_name)
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VALID_DIR = os.path.join(DATASET_DIR, "valid")
TRAIN_INPUT_DIR = os.path.join(TRAIN_DIR, "input")
TRAIN_OUTPUT_DIR = os.path.join(TRAIN_DIR, "output")
VALID_INPUT_DIR = os.path.join(VALID_DIR, "input")
VALID_OUTPUT_DIR = os.path.join(VALID_DIR, "output")
EX_TRAIN_INPUT_DIR = os.path.join(TRAIN_DIR, "ex_input")
EX_VALID_INPUT_DIR = os.path.join(VALID_DIR, "ex_input")

def make_file():
    # Ensure dataset directory exists
    os.makedirs(DATASET_DIR, exist_ok=True)

    # train directory
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TRAIN_INPUT_DIR, exist_ok=True)
    os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)
    os.makedirs(EX_TRAIN_INPUT_DIR, exist_ok=True)

    # valid directory
    os.makedirs(VALID_DIR, exist_ok=True)
    os.makedirs(VALID_INPUT_DIR, exist_ok=True)
    os.makedirs(VALID_OUTPUT_DIR, exist_ok=True)
    os.makedirs(EX_VALID_INPUT_DIR, exist_ok=True)

    for i in range(12):
        # os.makedirs(os.path.join(TRAIN_INPUT_DIR, str(i)), exist_ok=True)
        # os.makedirs(os.path.join(TRAIN_OUTPUT_DIR, str(i)), exist_ok=True)
        # os.makedirs(os.path.join(EX_TRAIN_INPUT_DIR, str(i)), exist_ok=True)
        os.makedirs(os.path.join(VALID_INPUT_DIR, str(i)), exist_ok=True)
        os.makedirs(os.path.join(VALID_OUTPUT_DIR, str(i)), exist_ok=True)
        os.makedirs(os.path.join(EX_VALID_INPUT_DIR, str(i)), exist_ok=True)

    print("Directory created.")

def make_csvfile(dir_path, file_num):
    if "train" in dir_path:
        csv_path = os.path.join(dir_path, 'train.csv')
    elif "valid" in dir_path:
        csv_path = os.path.join(dir_path, 'valid.csv')
    else:
        print("ディレクトリ名が不正です。")
        exit()

    data = []
    data.append(['x:in', 'y:out'])

    for i in range(file_num):
        data.append([f'./input/{i}.jpg', f'./output/{i}.jpg'])

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)

def make_csvfile2(dir_path, file_num):
    if "train" in dir_path:
        csv_path = os.path.join(dir_path, 'ex_train.csv')
    elif "valid" in dir_path:
        csv_path = os.path.join(dir_path, 'ex_valid.csv')
    else:
        print("ディレクトリ名が不正です。")
        exit()

    data = []
    data.append(['x:in', 'y:out'])

    for i in range(file_num):
        data.append([f'./ex_input/{i}.jpg', f'./output/{i}.jpg'])

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)

def set_color(hex_code):
    """
    16進数の色を引数にとる
    Parameters
    ----------
    hex_code : str
        16進数の色コード
    """
    # 16進数の色をRGBに変換
    r = hex_code[0]
    g = hex_code[1]
    b = hex_code[2]
    a = hex_code[3]

    # color
    bpy.data.materials["Material.001"].node_tree.nodes["Principled BSDF"].inputs[0].default_value = (r, g, b, a)

def init_camera():
    # setting resolution
    bpy.context.scene.render.resolution_x = IMG_SIZE[0]
    bpy.context.scene.render.resolution_y = IMG_SIZE[1]

def set_camera():
    # Randomize object settings
    x = random.uniform(16.0396, 18.3958)
    y = random.uniform(0, 6.26573)
    z = random.uniform(3.82227, 6.10865)


    bpy.data.objects['Empty'].rotation_euler[2] = z - random.uniform(0, 1) #random.uniform(0, 6.26573)
    bpy.data.objects['Light'].rotation_euler[2] = z - random.uniform(0, 1) #random.uniform(0, 6.26573)
    bpy.context.object.data.energy = random.uniform(10, 40)

    bpy.data.objects['Camera'].rotation_euler[0] = random.uniform(16.0396, 18.3958)
    bpy.data.objects['Camera'].rotation_euler[1] = random.uniform(0, 6.26573)
    bpy.data.objects['Camera'].rotation_euler[2] = z #random.uniform(3.82227, 6.10865)

    bpy.data.objects['S-IV b'].location[0] = random.uniform(260, 279)
    bpy.data.objects['S-IV b'].location[1] = random.uniform(40.447, 45.480)
    bpy.data.objects['S-IV b'].location[2] = random.uniform(-5, 7)

    bpy.data.objects['S-IV b'].rotation_euler[0] = random.uniform(0, 6.26573)
    bpy.data.objects['S-IV b'].rotation_euler[1] = random.uniform(0, 6.26573)
    bpy.data.objects['S-IV b'].rotation_euler[2] = random.uniform(0, 6.26573)

def calc_img_ratio(img):
    # Calculate the black ratio of the image

    # Convert image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate the black ratio
    total_pixels = gray_img.size
    black_mask = (gray_img == 0).astype(np.uint8)  # 黒色のピクセルを選択
    black_pixels = np.count_nonzero(black_mask)
    black_ratio = black_pixels / total_pixels

    # round black ratio
    black_ratio = round(black_ratio, 1)

    return black_ratio

def calc_iou(img1, img2):
    # Calculate the IoU of the image

    # Convert image to grayscale
    # gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Convert image to binary
    _, bin_img1 = cv2.threshold(img1, 1, 255, cv2.THRESH_BINARY)
    _, bin_img2 = cv2.threshold(img2, 1, 255, cv2.THRESH_BINARY)

    # divide by 255
    bin_img1 = bin_img1 / 255
    bin_img2 = bin_img2 / 255

    # save binary image
    # cv2.imwrite(os.path.join(TRAIN_INPUT_DIR, "a.jpg"), bin_img1)
    # cv2.imwrite(os.path.join(TRAIN_OUTPUT_DIR, "b.jpg"), bin_img2)
    # print(bin_img1)
    # print(bin_img2)
    # Calculate the IoU of white area
    intersection = np.logical_and(bin_img1, bin_img2)
    # union = np.logical_or(bin_img1, bin_img2)
    # iou = np.sum(intersection) / np.sum(union)
    intersection = np.sum(intersection) / np.sum(bin_img2)
    # if intersection != 0 and intersection != 1:
    #     print("break")
    #     print(bin_img1)
    #     print(bin_img2)
    #     # intersection = np.sum(intersection) / np.sum(bin_img2)
    #     print(intersection)
    #     time.sleep(4)
    # intersection = np.sum(intersection) / np.sum(bin_img2)

    intersection = round(intersection, 1)
    # print(intersection)
    # if intersection != 0 and intersection != 1:
    #     time.sleep(1)

    return intersection

def main():

    # make dataset directory
    make_file()

    # make csv file
    make_csvfile(TRAIN_DIR, TRAIN_PIC)
    make_csvfile(VALID_DIR, VALID_PIC)
    make_csvfile2(TRAIN_DIR, TRAIN_PIC)
    make_csvfile2(VALID_DIR, VALID_PIC)
    # make_csvfile(EX_TRAIN_INPUT_DIR, TRAIN_PIC)
    # make_csvfile(EX_VALID_INPUT_DIR, VALID_PIC)

    # if flug == 0 -> not save
    # if flug == 1 -> save
    
    pic_count_list = [0] * 11
    count = 0
    set_color(WHITE)

    # initialize camera
    init_camera()

    # loop to generate train images
    while True:
        print(pic_count_list)

        # if all pictures are generated, break
        if sum(pic_count_list) == VALID_PIC:
            break
    
        # set camera
        set_camera()

        # take image
        # Show the 'land_ocean_ice_cloud_8192' object for rendering
        bpy.data.objects["land_ocean_ice_cloud_8192"].hide_render = False
        bpy.data.objects["S-IV b"].hide_render = True

        earth_img = np.uint16(bpycv.render_data()["depth"] * 100000)
        print(type(earth_img))

        bpy.data.objects["land_ocean_ice_cloud_8192"].hide_render = True
        bpy.data.objects["S-IV b"].hide_render = False

        debris_img = np.uint16(bpycv.render_data()["depth"] * 1000)
        print(type(debris_img))

        intersection = calc_iou(earth_img, debris_img)

        print(intersection)

        if intersection == 0 and pic_count_list[0] < VALID_PIC/11:
            save_path1 = os.path.join(VALID_INPUT_DIR, "0",f"{pic_count_list[0]}.png")
            save_path2 = os.path.join(EX_VALID_INPUT_DIR, "0",f"{pic_count_list[0]}.png")
            save_path3 = os.path.join(VALID_OUTPUT_DIR, "0", f"{pic_count_list[0]}.png")
            pic_count_list[0] += 1

            # Save RGB image
            # set color to white
            bpy.data.objects["land_ocean_ice_cloud_8192"].hide_render = False
            # bpy.data.objects["S-IV b"].hide_render = False

            set_color(WHITE)
            result = bpycv.render_data()
            cv2.imwrite(save_path1, result["image"][..., ::-1])

            # set color to excolor
            set_color(EXCOLOR)
            result = bpycv.render_data()
            cv2.imwrite(save_path2, result["image"][..., ::-1])

            # Hide 'land_ocean_ice_cloud_8192' object for depth rendering
            bpy.data.objects["land_ocean_ice_cloud_8192"].hide_render = True

            # Render depth
            result = bpycv.render_data()

            # Save depth image
            depth_in_mm = result["depth"] * 1000
            _, bin_dt_img = cv2.threshold(depth_in_mm, 1, 255, cv2.THRESH_BINARY)
            cv2.imwrite(save_path3, depth_in_mm)
        
        # elif intersection == 1 and pic_count_list[10] < VALID_PIC/10:
        #     save_path1 = os.path.join(VALID_INPUT_DIR, "11",f"{pic_count_list[11]}.png")
        #     save_path2 = os.path.join(EX_VALID_INPUT_DIR, "11",f"{pic_count_list[11]}.png")
        #     save_path3 = os.path.join(VALID_OUTPUT_DIR, "11", f"{pic_count_list[11]}.png")
        #     pic_count_list[11] += 1

        #     # Save RGB image
        #     # set color to white
        #     bpy.data.objects["land_ocean_ice_cloud_8192"].hide_render = False
        #     # bpy.data.objects["S-IV b"].hide_render = False

        #     set_color(WHITE)
        #     result = bpycv.render_data()
        #     cv2.imwrite(save_path1, result["image"][..., ::-1])

        #     # set color to excolor
        #     set_color(EXCOLOR)
        #     result = bpycv.render_data()
        #     cv2.imwrite(save_path2, result["image"][..., ::-1])

        #     # Hide 'land_ocean_ice_cloud_8192' object for depth rendering
        #     bpy.data.objects["land_ocean_ice_cloud_8192"].hide_render = True

        #     # Render depth
        #     result = bpycv.render_data()

        #     # Save depth image
        #     depth_in_mm = result["depth"] * 1000
        #     _, bin_dt_img = cv2.threshold(depth_in_mm, 1, 255, cv2.THRESH_BINARY)
        #     cv2.imwrite(save_path3, depth_in_mm)

        else:
            if pic_count_list[int(intersection*10)] < VALID_PIC/11:
                save_path1 = os.path.join(VALID_INPUT_DIR, str(int(intersection*10)),f"{pic_count_list[int(intersection*10)]}.png")
                save_path2 = os.path.join(EX_VALID_INPUT_DIR, str(int(intersection*10)),f"{pic_count_list[int(intersection*10)]}.png")
                save_path3 = os.path.join(VALID_OUTPUT_DIR, str(int(intersection*10)), f"{pic_count_list[int(intersection*10)]}.png")
                pic_count_list[int(intersection*10)] += 1

                # Save RGB image
                # set color to white
                bpy.data.objects["land_ocean_ice_cloud_8192"].hide_render = False
                # bpy.data.objects["S-IV b"].hide_render = False

                set_color(WHITE)
                result = bpycv.render_data()
                cv2.imwrite(save_path1, result["image"][..., ::-1])

                # set color to excolor
                set_color(EXCOLOR)
                result = bpycv.render_data()
                cv2.imwrite(save_path2, result["image"][..., ::-1])

                # Hide 'land_ocean_ice_cloud_8192' object for depth rendering
                bpy.data.objects["land_ocean_ice_cloud_8192"].hide_render = True

                # Render depth
                result = bpycv.render_data()

                # Save depth image
                depth_in_mm = result["depth"] * 1000
                _, bin_dt_img = cv2.threshold(depth_in_mm, 1, 255, cv2.THRESH_BINARY)
                cv2.imwrite(save_path3, depth_in_mm)
            
            if sum(pic_count_list) == VALID_PIC:
                break
        
        # if intersection == 0 or intersection == 1:
        #     if pic_count_list[int(-intersection)] < VALID_PIC/10:
        #         if intersection == 0:
        #             dir_num = 0
        #         else:
        #             dir_num = 11
        #         save_path1 = os.path.join(VALID_INPUT_DIR, str(dir_num),f"{pic_count_list[int(-intersection)]}.png")
        #         save_path2 = os.path.join(EX_VALID_INPUT_DIR, str(dir_num),f"{pic_count_list[int(-intersection)]}.png")
        #         save_path3 = os.path.join(VALID_OUTPUT_DIR, str(dir_num), f"{pic_count_list[int(-intersection)]}.png")
        #         pic_count_list[int(-intersection)] += 1
        #         flug = 1
        # else:
        #     if pic_count_list[int(intersection*10)] < VALID_PIC/10:
        #         save_path1 = os.path.join(VALID_INPUT_DIR, str(int(intersection*10)),f"{pic_count_list[int(intersection*10)]}.png")
        #         save_path2 = os.path.join(EX_VALID_INPUT_DIR, str(int(intersection*10)),f"{pic_count_list[int(intersection*10)]}.png")
        #         save_path3 = os.path.join(VALID_OUTPUT_DIR, str(int(intersection*10)), f"{pic_count_list[int(intersection*10)]}.png")
        #         pic_count_list[int(intersection*10)] += 1
        #         flug = 1

        # if flug == 1:
        #     # Save RGB image
        #     # set color to white
        #     bpy.data.objects["land_ocean_ice_cloud_8192"].hide_render = False
        #     # bpy.data.objects["S-IV b"].hide_render = False

        #     set_color(WHITE)
        #     result = bpycv.render_data()
        #     cv2.imwrite(save_path1, result["image"][..., ::-1])

        #     # set color to excolor
        #     set_color(EXCOLOR)
        #     result = bpycv.render_data()
        #     cv2.imwrite(save_path2, result["image"][..., ::-1])

        #     # Hide 'land_ocean_ice_cloud_8192' object for depth rendering
        #     bpy.data.objects["land_ocean_ice_cloud_8192"].hide_render = True

        #     # Render depth
        #     result = bpycv.render_data()

        #     # Save depth image
        #     depth_in_mm = result["depth"] * 1000
        #     cv2.imwrite(save_path3, np.uint16(depth_in_mm))
        

        # count += 1

    #     if intersection == 0:
    #         set_color(WHITE)
    #         set_color(WHITE)
    #         result = bpycv.render_data()
    #         cv2.imwrite(os.path.join(TRAIN_INPUT_DIR, f"{count}.jpg"), result["image"][..., ::-1])


    #     # Render image, instance annotation, and depth
    #     result = bpycv.render_data()

    #     black_ratio = calc_img_ratio(result["image"])

    #     if black_ratio == 1.0:
    #         flug = 0
    #     else:
    #         if pic_count_list[int(black_ratio*10)] < TRAIN_PIC/10:
    #             pic_count_list[int(black_ratio*10)] += 1
    #             flug = 1
    #         else:
    #             flug = 0

    #     # save RGB image and depth image
    #     if flug == 1:
    #         # Save RGB image
    #         # set color to white
    #         set_color(WHITE)
    #         result = bpycv.render_data()
    #         cv2.imwrite(os.path.join(TRAIN_INPUT_DIR, f"{count}.jpg"), result["image"][..., ::-1])

    #         # set color to excolor
    #         set_color(EXCOLOR)
    #         result = bpycv.render_data()
    #         cv2.imwrite(os.path.join(EX_TRAIN_INPUT_DIR, f"{count}.jpg"), result["image"][..., ::-1])

    #         # Hide 'land_ocean_ice_cloud_8192' object for depth rendering
    #         bpy.data.objects["land_ocean_ice_cloud_8192"].hide_render = True

    #         # Render depth
    #         result = bpycv.render_data()

    #         # Save depth image
    #         depth_in_mm = result["depth"] * 1000
    #         cv2.imwrite(os.path.join(TRAIN_OUTPUT_DIR, f"{count}.jpg"), np.uint16(depth_in_mm))

    #         count += 1

    # # loop to generate valid images
    # pic_count_list = [0] * 10
    # count = 0
    # set_color(WHITE)

    # while True:
    #     print(pic_count_list)

    #     # if all pictures are generated, break
    #     if sum(pic_count_list) == VALID_PIC:
    #         break
    
    #     # set camera
    #     set_camera()

    #     # take image
    #     # Show the 'land_ocean_ice_cloud_8192' object for rendering
    #     bpy.data.objects["land_ocean_ice_cloud_8192"].hide_render = False

    #     # Render image, instance annotation, and depth
    #     result = bpycv.render_data()

    #     black_ratio = calc_img_ratio(result["image"])

    #     if black_ratio == 1.0:
    #         flug = 0
    #     else:
    #         if pic_count_list[int(black_ratio*10)] < VALID_PIC/10:
    #             pic_count_list[int(black_ratio*10)] += 1
    #             flug = 1
    #         else:
    #             flug = 0

    #     # save RGB image and depth image
    #     if flug == 1:
    #         # Save RGB image
    #         # set color to white
    #         set_color(WHITE)
    #         result = bpycv.render_data()
    #         cv2.imwrite(os.path.join(VALID_INPUT_DIR, f"{count}.jpg"), result["image"][..., ::-1])

    #         # set color to excolor
    #         set_color(EXCOLOR)
    #         result = bpycv.render_data()
    #         cv2.imwrite(os.path.join(EX_VALID_INPUT_DIR, f"{count}.jpg"), result["image"][..., ::-1])

    #         # Hide 'land_ocean_ice_cloud_8192' object for depth rendering
    #         bpy.data.objects["land_ocean_ice_cloud_8192"].hide_render = True

    #         # Render depth
    #         result = bpycv.render_data()

    #         # Save depth image
    #         depth_in_mm = result["depth"] * 1000
    #         cv2.imwrite(os.path.join(VALID_OUTPUT_DIR, f"{count}.jpg"), np.uint16(depth_in_mm))

    #         count += 1

if __name__ == "__main__":
    
    start_time = time.time()
    main()

    elapsed_time = time.time() - start_time
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")