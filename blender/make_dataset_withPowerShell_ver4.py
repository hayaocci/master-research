# PowerShellで実行するためのスクリプト

import os
import numpy as np
# import cv2
import random
import datetime as dt
import bpy
# import bpycv
import csv
import time

# Constants
IMG_SIZE = (512, 512)
LABEL_IMG_SIZE = (512, 512)
NUM_PIC = 10000 # 100~ && x100
SPLIT_RATIO = 0.8
# TRAIN_PIC = int(NUM_PIC * SPLIT_RATIO)
# TRAIN_PIC = 8000
TRAIN_PIC = int(NUM_PIC * SPLIT_RATIO)
# VALID_PIC = 11*200 #NUM_PIC - TRAIN_PIC
VALID_PIC = int(NUM_PIC - TRAIN_PIC)

if NUM_PIC > 500:
    BATCH_SIZE = 500
else:
    BATCH_SIZE = NUM_PIC

RANDOM_SETTING = False # True: randomize camera location, False: fixed camera location

# directory name manual
dir_name = "20240812_2_" + str(IMG_SIZE[0]) + "x" + str(IMG_SIZE[1])

# path
# SAVE_DIR = 'C:/workspace/senior_thesis/nnc001/dataset/'
SAVE_DIR = 'C:/workspace/MasterResearch/blender_dataset'
DATASET_DIR = os.path.join(SAVE_DIR, dir_name)
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VALID_DIR = os.path.join(DATASET_DIR, "valid")
TRAIN_INPUT_DIR = os.path.join(TRAIN_DIR, "input")
TRAIN_OUTPUT_DIR = os.path.join(TRAIN_DIR, "output")
VALID_INPUT_DIR = os.path.join(VALID_DIR, "input")
VALID_OUTPUT_DIR = os.path.join(VALID_DIR, "output")
# EX_TRAIN_INPUT_DIR = os.path.join(TRAIN_DIR, "ex_input")
# EX_VALID_INPUT_DIR = os.path.join(VALID_DIR, "ex_input")

BLENDER_FILEPATH = 'C:/workspace/MasterResearch/blender/new_earth_ver1.10_scripting_withCamera_sun_synchronous/new_earth/sun_synchronous_orbit1.10.blend'

def make_dir():
    """
    データセット保存用のディレクトリを作成する関数
    """
    os.makedirs(DATASET_DIR, exist_ok=True)
    # train directory
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TRAIN_INPUT_DIR, exist_ok=True)
    os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)
    # valid directory
    os.makedirs(VALID_DIR, exist_ok=True)
    os.makedirs(VALID_INPUT_DIR, exist_ok=True)
    os.makedirs(VALID_OUTPUT_DIR, exist_ok=True)

    print("Directory created.")

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

def init_camera():
    # setting resolution
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.render.resolution_x = IMG_SIZE[0]
    bpy.context.scene.render.resolution_y = IMG_SIZE[1]
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 64

def init_camera_small():
    # setting resolution
    bpy.context.scene.cycles.use_denoising = False
    bpy.context.scene.render.resolution_x = LABEL_IMG_SIZE[0]
    bpy.context.scene.render.resolution_y = LABEL_IMG_SIZE[1]
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 64

def init_gpu():
    bpy.context.preferences.system.memory_cache_limit = 2  # 2GBのメモリキャッシュ制限
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    bpy.context.scene.cycles.device = 'GPU'

def purge_orphan_data():
    # Purge orphan data to free memory
    for _ in range(3):  # Repeat to ensure all orphans are purged
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)

def set_all_object_old():
    # sun light settings
    bpy.data.objects['Sun'].rotation_euler[2] = random.uniform(-3.64774, 0.506145)
    # earth settings
    random_earth = random.uniform(0, 6.26573)
    bpy.data.objects['earth'].rotation_euler[2] = random_earth
    bpy.data.objects['atmosphere'].rotation_euler[2] = random_earth
    bpy.data.objects['cloud'].rotation_euler[2] = random_earth
    # rabdomize object settings
    # rotation
    random_rotx = random.uniform(0, 6.26573)
    random_roty = random.uniform(0, 6.26573)
    random_rotz = random.uniform(0, 6.26573)

    bpy.data.objects['debris'].rotation_euler[0] = random_rotx
    bpy.data.objects['debris'].rotation_euler[1] = random_roty
    bpy.data.objects['debris'].rotation_euler[2] = random_rotz

    bpy.data.objects['decoydebris'].rotation_euler[0] = random_rotx
    bpy.data.objects['decoydebris'].rotation_euler[1] = random_roty
    bpy.data.objects['decoydebris'].rotation_euler[2] = random_rotz

    # camera settings
    # rotation
    bpy.data.objects['CameraEmpty'].rotation_euler[0] = random.uniform(0, 6.26573)
    bpy.data.objects['CameraEmpty'].rotation_euler[1] = random.uniform(0, 6.26573)
    bpy.data.objects['CameraEmpty'].rotation_euler[2] = random.uniform(0, 6.26573)
    # location
    if RANDOM_SETTING == True:
        random_y = random.uniform(680, 740)

        # location xz
        if random_y >= 730:
            x = 17
            z = 17
        elif random_y > 720 and random_y < 730:
            x = 15
            z = 15
        elif random_y > 710 and random_y <= 720:
            x = 12
            z = 12
        elif random_y > 700 and random_y <= 710:
            x = 8
            z = 8
        elif random_y > 690 and random_y <= 700:
            x = 5
            z = 5
        elif random_y > 680 and random_y <= 690:
            x = 2
            z = 2

        random_x = random.uniform(-x, x)
        random_z = random.uniform(-z, z)
    else:
        random_y = 685
        random_x = 0
        random_z = 0

    bpy.data.objects['Camera'].location[0] = random_x
    bpy.data.objects['Camera'].location[1] = random_y
    bpy.data.objects['Camera'].location[2] = random_z

    # cam rotation
    bpy.data.objects['Camera'].rotation_euler[0] = random.uniform(-1.65806, -1.39626)
    bpy.data.objects['Camera'].rotation_euler[1] = random.uniform(0, 6.26573)
    bpy.data.objects['Camera'].rotation_euler[2] = random.uniform(-0.174533, 0.174533)

def set_object_sun_synchronous():
    # earth settings
    random_earth = random.uniform(0, 6.26573)
    bpy.data.objects['earth'].rotation_euler[2] = random_earth
    bpy.data.objects['atmosphere'].rotation_euler[2] = random_earth
    bpy.data.objects['cloud'].rotation_euler[2] = random_earth

    # rabdomize debris settings
    # rotation
    random_rotx = random.uniform(1.39626, 1.74533) # 80~100 deg
    random_roty = random.uniform(0, 6.26573) # 0~360 deg
    random_rotz = random.uniform(-0.174533, 0.174533) # -10~10 deg
    bpy.data.objects['debris'].rotation_euler[0] = random_rotx
    bpy.data.objects['debris'].rotation_euler[1] = random_roty
    bpy.data.objects['debris'].rotation_euler[2] = random_rotz
    bpy.data.objects['decoydebris'].rotation_euler[0] = random_rotx
    bpy.data.objects['decoydebris'].rotation_euler[1] = random_roty
    bpy.data.objects['decoydebris'].rotation_euler[2] = random_rotz

    # camera settings
    # rotation
    bpy.data.objects['CameraEmpty'].rotation_euler[0] = 0 # 0固定
    bpy.data.objects['CameraEmpty'].rotation_euler[1] = 0.340688 # 19.52 deg 固定
    bpy.data.objects['CameraEmpty'].rotation_euler[2] = 4.9707 # 284.8 deg 固定

    # location
    # bpy.data.objects['DebrisCenter'].rotation_euler[0] = random.uniform(-0.5, 1.3)
    bpy.data.objects['DebrisCenter'].rotation_euler[0] = random.uniform(0.3, 3.5)
    # bpy.data.objects['DebrisCenter'].rotation_euler[0] = random.uniform(-0.99, 1.85)
    bpy.data.objects['DebrisCenter'].rotation_euler[1] = 0 #固定
    bpy.data.objects['DebrisCenter'].rotation_euler[2] = 1.5708 #固定

def init_camera_location():
    """
    地球が写りこむようにカメラを配置する関数
    """
    bpy.data.objects['Camera'].rotation[0] = 3.4034
    bpy.data.objects['Camera'].rotation[1] = 1.22173
    bpy.data.objects['Camera'].rotation[2] = 0


def render_img(input_dir, output_dir, file_count):
    # Render the image
    bpy.data.objects["atmosphere"].hide_render = False
    bpy.data.objects["cloud"].hide_render = False
    bpy.data.objects["earth"].hide_render = False
    
    bpy.data.objects["debris"].hide_render = False
    bpy.data.objects["decoydebris"].hide_render = True

    bpy.context.scene.render.filepath = input_dir + "/" + file_count + ".png"
    bpy.context.scene.render.image_settings.file_format = 'PNG' 
    bpy.ops.render.render(write_still=True) 

    init_camera_small()
    # Render the Mask
    bpy.data.objects["atmosphere"].hide_render = True
    bpy.data.objects["cloud"].hide_render = True
    bpy.data.objects["earth"].hide_render = True

    bpy.data.objects["debris"].hide_render = True
    bpy.data.objects["decoydebris"].hide_render = False

    bpy.context.scene.render.filepath = output_dir + "/" + file_count + ".png"
    bpy.context.scene.render.image_settings.file_format = 'PNG' 
    bpy.ops.render.render(write_still=True) 
    
    purge_orphan_data()



def render_batch(start_idx, end_idx, input_dir, output_dir):
    make_dir()

    # init_camera()

    for i in range(start_idx, end_idx):
        
        try:
            init_camera()
            # sun light settings
            bpy.data.objects['Sun'].rotation_euler[2] = random.uniform(-3.64774, 0.506145)

            # earth settings
            random_earth = random.uniform(0, 6.26573)
            bpy.data.objects['earth'].rotation_euler[2] = random_earth
            bpy.data.objects['atmosphere'].rotation_euler[2] = random_earth
            bpy.data.objects['cloud'].rotation_euler[2] = random_earth

            # rabdomize object settings
            # rotation
            random_rotx = random.uniform(0, 6.26573)
            random_roty = random.uniform(0, 6.26573)
            random_rotz = random.uniform(0, 6.26573)

            bpy.data.objects['debris'].rotation_euler[0] = random_rotx
            bpy.data.objects['debris'].rotation_euler[1] = random_roty
            bpy.data.objects['debris'].rotation_euler[2] = random_rotz

            bpy.data.objects['decoydebris'].rotation_euler[0] = random_rotx
            bpy.data.objects['decoydebris'].rotation_euler[1] = random_roty
            bpy.data.objects['decoydebris'].rotation_euler[2] = random_rotz

            # camera settings
            # rotation
            bpy.data.objects['CameraEmpty'].rotation_euler[0] = random.uniform(0, 6.26573)
            bpy.data.objects['CameraEmpty'].rotation_euler[1] = random.uniform(0, 6.26573)
            bpy.data.objects['CameraEmpty'].rotation_euler[2] = random.uniform(0, 6.26573)

            # location
            if RANDOM_SETTING == True:
                random_y = random.uniform(680, 740)

                # location xz
                if random_y >= 730:
                    x = 17
                    z = 17
                elif random_y > 720 and random_y < 730:
                    x = 15
                    z = 15
                elif random_y > 710 and random_y <= 720:
                    x = 12
                    z = 12
                elif random_y > 700 and random_y <= 710:
                    x = 8
                    z = 8
                elif random_y > 690 and random_y <= 700:
                    x = 5
                    z = 5
                elif random_y > 680 and random_y <= 690:
                    x = 2
                    z = 2

                random_x = random.uniform(-x, x)
                random_z = random.uniform(-z, z)
            else:
                random_y = 685
                random_x = 0
                random_z = 0

            bpy.data.objects['Camera'].location[0] = random_x
            bpy.data.objects['Camera'].location[1] = random_y
            bpy.data.objects['Camera'].location[2] = random_z

            # cam rotation
            bpy.data.objects['Camera'].rotation_euler[0] = random.uniform(-1.65806, -1.39626)
            bpy.data.objects['Camera'].rotation_euler[1] = random.uniform(0, 6.26573)
            bpy.data.objects['Camera'].rotation_euler[2] = random.uniform(-0.174533, 0.174533)

            # Render the image
            bpy.data.objects["atmosphere"].hide_render = False
            bpy.data.objects["cloud"].hide_render = False
            bpy.data.objects["earth"].hide_render = False
            
            bpy.data.objects["debris"].hide_render = False
            bpy.data.objects["decoydebris"].hide_render = True

            bpy.context.scene.render.filepath = input_dir + "/" + str(i) + ".png"
            bpy.context.scene.render.image_settings.file_format = 'PNG' 
            bpy.ops.render.render(write_still=True) 

            init_camera_small()
            # Render the Mask
            bpy.data.objects["atmosphere"].hide_render = True
            bpy.data.objects["cloud"].hide_render = True
            bpy.data.objects["earth"].hide_render = True

            bpy.data.objects["debris"].hide_render = True
            bpy.data.objects["decoydebris"].hide_render = False

            bpy.context.scene.render.filepath = output_dir + "/" + str(i) + ".png"
            bpy.context.scene.render.image_settings.file_format = 'PNG' 
            bpy.ops.render.render(write_still=True) 
            
            purge_orphan_data()

        except Exception as e:
            time.sleep(60)

def render_main():
    make_dir()

    isTrainCompleted = False

    if get_file_num(TRAIN_OUTPUT_DIR) == TRAIN_PIC:
        isTrainCompleted = True
        pass
    else:
        # render
        file_count = get_file_num(TRAIN_OUTPUT_DIR)
        
        # init_camera_location()
        for i in range(BATCH_SIZE):
            try:
                # init_camera()
                set_all_object()
                # set_object_sun_synchronous()

                render_img(TRAIN_INPUT_DIR, TRAIN_OUTPUT_DIR, str(file_count+i))

            except Exception as e:
                print("Error: ", e)
                time.sleep(60)

            if file_count+i == TRAIN_PIC:
                isTrainCompleted = True
                break
        # bpy.ops.wm.read_factory_settings(use_empty=True)
        # bpy.ops.wm.open_mainfile(filepath=BLENDER_FILEPATH)

    if isTrainCompleted == True:
        if get_file_num(VALID_OUTPUT_DIR) == VALID_PIC:
            pass
        else:
            # render
            file_count = get_file_num(VALID_OUTPUT_DIR)

            # init_camera_location()
            for i in range(BATCH_SIZE):
                try:
                    # init_camera()
                    set_all_object()
                    # set_object_sun_synchronous()

                    render_img(VALID_INPUT_DIR, VALID_OUTPUT_DIR, str(file_count+i))

                except Exception as e:
                    print("Error: ", e)
                    time.sleep(60)

                if file_count+i == VALID_PIC:
                    break
            # bpy.ops.wm.read_factory_settings(use_empty=True)
            # bpy.ops.wm.open_mainfile(filepath=BLENDER_FILEPATH)
            
    print("Rendering Completed")

def render_main_old():
    make_dir()

    # get file count
    file_count = get_file_num(TRAIN_INPUT_DIR)

    # init_gpu()

    for batch_start in range(0, TRAIN_PIC, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, TRAIN_PIC)
        # render_batch(batch_start, batch_end, os.path.join(TRAIN_INPUT_DIR, str(int(batch_start/BATCH_SIZE))), os.path.join(TRAIN_OUTPUT_DIR, str(int(batch_start/BATCH_SIZE))))
        render_batch(batch_start, batch_end, TRAIN_INPUT_DIR, TRAIN_OUTPUT_DIR)
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.wm.open_mainfile(filepath=BLENDER_FILEPATH)
        # init_gpu()

    for batch_start in range(0, VALID_PIC, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, VALID_PIC)
        # render_batch(batch_start, batch_end, os.path.join(VALID_INPUT_DIR, str(int(batch_start/BATCH_SIZE))), os.path.join(VALID_OUTPUT_DIR, str(int(batch_start/BATCH_SIZE))))
        render_batch(batch_start, batch_end, VALID_INPUT_DIR, VALID_OUTPUT_DIR)
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.wm.open_mainfile(filepath=BLENDER_FILEPATH)
        # init_gpu()

def random_parameter():
    debrisCenterRotZ = random.uniform(-1.0472, 1.0472) # -60~60 deg

    if debrisCenterRotZ > 0:
        cameraEmptyLocX = random.uniform(0, 0.436332) # 0~25 deg
    else:
        cameraEmptyLocX = random.uniform(-0.46332, 0) # -25~0 deg

    cameraEmptyLocY = 100
    cameraEmptyLocZ = random.uniform(-0.46332, 0.436332) # -25~25 deg

    cameraEmptyRotX = random.uniform(-0.244346, 0.244346) # -14~14 deg
    cameraEmptyRotZ = random.uniform(-0.244346, 0.244346) # -14~14 deg

    debrisRotZ = -debrisCenterRotZ

    parameter_list = [debrisCenterRotZ, cameraEmptyLocX, cameraEmptyLocY, cameraEmptyLocZ, cameraEmptyRotX, cameraEmptyRotZ, debrisRotZ]

    return parameter_list

def set_all_object():
    # sun light settings
    # bpy.data.objects['Sun'].rotation_euler[2] = random.uniform(-3.64774, 0.506145)
    # bpy.data.objects['Sun'].rotation_euler[2] = random.uniform(-6.1283, 0.436332)
    # bpy.data.objects['Sun'].rotation_euler[2] = random.uniform(-3.00197, 0.226893)
    bpy.data.objects['Sun'].rotation_euler[2] = random.uniform(-2.98451, -0.261799)

    # earth settings
    random_earth = random.uniform(0, 6.26573)
    bpy.data.objects['earth'].rotation_euler[2] = random_earth
    bpy.data.objects['atmosphere'].rotation_euler[2] = random_earth
    bpy.data.objects['cloud'].rotation_euler[2] = random_earth

    parameter_list = random_parameter()

    # randomize DebrisCenter settings
    bpy.data.objects['DebrisCenter'].rotation_euler[2] = parameter_list[0]

    # randomize CameraEmpty settings
    bpy.data.objects['CameraEmpty'].location[0] = parameter_list[1]
    bpy.data.objects['CameraEmpty'].location[1] = parameter_list[2]
    bpy.data.objects['CameraEmpty'].location[2] = parameter_list[3]
    bpy.data.objects['CameraEmpty'].rotation_euler[0] = parameter_list[4]
    bpy.data.objects['CameraEmpty'].rotation_euler[2] = parameter_list[5]

    # randomize debris settings
    bpy.data.objects['debris'].rotation_euler[2] = parameter_list[6]
    bpy.data.objects['decoydebris'].rotation_euler[2] = parameter_list[6]




if __name__ == "__main__":
    
    start_time = time.time()

    render_main()

    elapsed_time = time.time() - start_time
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    bpy.ops.wm.quit_blender()
    # bpy.ops.wm.quit_blender({'CANCELLED'})