import cv2
import numpy as np

# def convert_image(input_image_path, output_image_path):
#     # 24ビットのRGB画像を読み込む
#     img = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    
#     if img is not None:
#         # 画像が3チャンネル(RGB)であることを確認
#         if len(img.shape) == 3:
#             # 黒以外のピクセルを白に変換
#             mask = cv2.inRange(img, (1, 1, 1), (255, 255, 255))
#             img[mask == 255] = [255, 255, 255]

#             # 8ビットの画像として保存
#             img_8bit = cv2.convertScaleAbs(img, alpha=(255.0/255.0))
#             cv2.imwrite(output_image_path, img_8bit)
#             print("Image processing complete. The output image is saved as", output_image_path)
#         else:
#             print("Error: Image is not in 24-bit RGB format.")
#     else:
#         print("Error: Image not loaded correctly.")

def convert_image(input_image_path, output_image_path):
    # 24ビットのRGB画像を読み込む
    img = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    
    if img is not None:
        # 画像が3チャンネル(RGB)であることを確認
        if len(img.shape) == 3:
            # 画像をグレイスケールに変換
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 黒以外のピクセルを白に変換
            _, thresholded_img = cv2.threshold(gray_img, 10, 255, cv2.THRESH_BINARY)
            
            # 変換後の画像を保存（ビット深さ8ビット）
            cv2.imwrite(output_image_path, thresholded_img)
            print("Image processing complete. The output image is saved as", output_image_path)
        else:
            print("Error: Image is not in 24-bit RGB format.")
    else:
        print("Error: Image not loaded correctly.")

# 入力画像と出力画像のパスを指定
# input_image_path = 'input_image.png'
# output_image_path = 'output_image.png'

# # 画像を変換
# convert_image(input_image_path, output_image_path)

for i in range(2000):
    input_image_path = f'20240624_dataset/valid/label/{i+8000}.png'
    output_image_path = f'20240624_dataset/valid/bin_label/{i+8000}.png'
    convert_image(input_image_path, output_image_path)
