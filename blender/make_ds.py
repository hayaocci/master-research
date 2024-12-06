# import os
# import numpy as np
# # import cv2
# import random
# import datetime as dt
# import bpy
# # import bpycv
# import csv
# import time

# START_ANGLE = 0
# AGNLE_STEP = 0.1
# END_ANGLE = 360

# IMG_SIZE = (224, 224)

# SAVE_PATH = "C:/workspace/MasterResearch/blender_dataset"
# DATASET_DIR = os.path.join(SAVE_PATH, "20241113_224x224")


# def init_camera():
#     # setting resolution
#     bpy.context.scene.cycles.use_denoising = True
#     bpy.context.scene.render.resolution_x = IMG_SIZE[0]
#     bpy.context.scene.render.resolution_y = IMG_SIZE[1]
#     bpy.context.scene.render.engine = 'CYCLES'
#     bpy.context.scene.cycles.samples = 64

# def init_gpu():
#     bpy.context.preferences.system.memory_cache_limit = 2  # 2GBのメモリキャッシュ制限
#     bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
#     bpy.context.scene.cycles.device = 'GPU'


# def main(input_dir):
    
#     # Initializations
#     init_camera()
#     init_gpu()


#     photo_angle = START_ANGLE

#     while True:

#         # render image
#         bpy.context.scene.render.filepath = input_dir + "/" + str(photo_angle) + ".png"
#         bpy.context.scene.render.image_settings.file_format = 'PNG' 
#         bpy.ops.render.render(write_still=True)
        
#         # change angle of sunlight
#         bpy.data.objects['Sun'].rotation_euler[2] = np.deg2rad(photo_angle)
#         # bpy.context.scene.render.filepath = f'./photos/{photo_angle}.png'
#         # bpy.ops.render.render(write_still=True)


# if __name__ == "__main__":
#     # input_dir = 'C:/workspace/MasterResearch/blender/'
#     main(DATASET_DIR)

import os
import numpy as np
import bpy

START_ANGLE = 0
ANGLE_STEP = 60
END_ANGLE = 360
IMG_SIZE = (224, 224)
SAVE_PATH = "C:/workspace/MasterResearch/blender_dataset"
DATASET_DIR = os.path.join(SAVE_PATH, "20241113_224x224")

ANGLE_RANGE = 6.28319  # 360 degrees in radians
ANGLE_PER_STEP = ANGLE_RANGE / 360

def init_camera():
    # Set camera resolution and other render settings
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.render.resolution_x = IMG_SIZE[0]
    bpy.context.scene.render.resolution_y = IMG_SIZE[1]
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 64

def init_gpu():
    # Set GPU settings
    bpy.context.preferences.system.memory_cache_limit = 2  # 2GB memory cache limit
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    bpy.context.scene.cycles.device = 'GPU'

def main(input_dir):
    # Initialize camera and GPU settings
    init_camera()
    init_gpu()

    # Iterate over angles from START_ANGLE to END_ANGLE with ANGLE_STEP
    for angle in np.arange(START_ANGLE, END_ANGLE, ANGLE_STEP):
        # Set render file path with the angle as the file name
        bpy.context.scene.render.filepath = os.path.join(input_dir, f"{angle:.1f}.png")
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        
        # Adjust sunlight angle
        bpy.data.objects['Sun'].rotation_euler[1] = np.deg2rad(angle)
        bpy.data.objects['Sun'].rotation_euler[1] = 

        # Render image
        bpy.ops.render.render(write_still=True)
        print(f"Rendered image at angle {angle:.1f}")

if __name__ == "__main__":
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
    main(DATASET_DIR)
