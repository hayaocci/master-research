import cv2
import numpy as np

def change_contrast(image, alpha, beta):
    # Change contrast
    new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return new_image

img = cv2.imread("../blender/dataset-example/20240623_134443_96x96/train/input/4.png")
new_img = change_contrast(img, 10, 0)
# cv2.imshow('Image', new_img)
cv2.imwrite('output.png', new_img)

def sigmoidTone(input_img, k=0.05, x0=127.5):
    output_float = 255 / (1 + np.exp(-k * (input_img - x0) ) )
    return output_float.astype(np.uint8)

def crop_square(image):
    # Crop image
    height, width = image.shape[:2]
    center_x = width // 2
    center_y = height // 2
    pixel = min(height, width)
    cropped_image = image[center_y - pixel // 2: center_y + pixel // 2, center_x - pixel // 2: center_x + pixel // 2]
    return cropped_image

def change_size(image: np.ndarray, size: tuple):
    # Change size
    new_image = cv2.resize(image, size)
    return new_image
