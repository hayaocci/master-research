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


