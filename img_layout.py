import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# 画像のパス（表示したい2枚に調整）
image_paths = ['real1.png', 'output1.png']

# 画像を読み込む
images = [mpimg.imread(img_path) for img_path in image_paths]

# 2番目の画像を2値化する（例として0.5をしきい値とする）
threshold = 0.5
images[1] = (images[1] > threshold).astype(np.float32)

# 横並びで画像を表示（2つに調整）
fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # サイズ調整

# 各画像をプロット
for i, (ax, img) in enumerate(zip(axes, images)):
    if i == 2:  # 2番目の画像の場合
        ax.imshow(img, cmap='viridis', extent=[0, img.shape[1], 0, img.shape[0]])  # 'viridis'カラーマップ
    else:
        ax.imshow(img, cmap='gray', extent=[0, img.shape[1], 0, img.shape[0]])  # グレースケール

    ax.axis('on')  # 軸を表示

    # 目盛りの最大値を表示
    ax.set_xticks([0, img.shape[1]])  # x軸の0と最大値を表示
    ax.set_yticks([0, img.shape[0]])  # y軸の0と最大値を表示

# 余白を調整
plt.subplots_adjust(wspace=0.1)  # 画像間の余白

# 画像を保存
plt.savefig('output_image_with_margin_and_axes_and_viridis_cmap.png', bbox_inches='tight', pad_inches=0.1)

# 表示
plt.show()


# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import numpy as np

# # 画像のパス
# image_paths = ['real.png', 'output.png', 'pred.png']

# # 画像を読み込む
# images = [mpimg.imread(img_path) for img_path in image_paths]

# # 真ん中の画像を2値化する (例として0.5をしきい値とする)
# threshold = 0.5
# images[1] = (images[1] > threshold).astype(np.float32)

# # 横並びで画像を表示
# fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # サイズ調整

# # 各画像をプロット
# for i, (ax, img) in enumerate(zip(axes, images)):
#     if i == 2:  # 3番目の画像の場合
#         ax.imshow(img, cmap='viridis', extent=[0, img.shape[1], 0, img.shape[0]])  # 'viridis'カラーマップ
#     else:
#         ax.imshow(img, cmap='gray', extent=[0, img.shape[1], 0, img.shape[0]])  # ピクセルサイズに合わせて表示
    
#     ax.axis('on')  # 軸を表示
    
#     # 目盛りの最大値を表示
#     ax.set_xticks([0, img.shape[1]])  # x軸の0と最大値を表示
#     ax.set_yticks([0, img.shape[0]])  # y軸の0と最大値を表示

# # 余白を調整
# plt.subplots_adjust(wspace=0.1)  # 画像間の余白

# # 画像を保存
# plt.savefig('output_image_with_margin_and_axes_and_viridis_cmap.png', bbox_inches='tight', pad_inches=0.1)

# # 表示
# plt.show()

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# 画像のパス（表示したい2枚に調整）
image_paths = ['real1.png', 'output1.png']

# 画像を読み込む
images = [mpimg.imread(img_path) for img_path in image_paths]

# 2番目の画像を2値化する（例として0.5をしきい値とする）
threshold = 0.5
images[1] = (images[1] > threshold).astype(np.float32)

# 横並びで画像を表示（2つに調整）
fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # サイズ調整

# 各画像をプロット
for i, (ax, img) in enumerate(zip(axes, images)):
    if i == 3:  # 2番目の画像の場合
        im = ax.imshow(img, cmap='viridis', extent=[0, img.shape[1], 0, img.shape[0]])  # 'viridis'カラーマップ
    else:
        im = ax.imshow(img, cmap='gray', extent=[0, img.shape[1], 0, img.shape[0]])  # グレースケール
    
    ax.axis('on')  # 軸を表示

    # 目盛りの最大値を表示
    ax.set_xticks([0, img.shape[1]])  # x軸の0と最大値を表示
    ax.set_yticks([0, img.shape[0]])  # y軸の0と最大値を表示

    # 軸の数値（目盛りラベル）のフォントサイズを大きくする
    ax.tick_params(axis='both', which='major', labelsize=24)  # フォントサイズを14に設定

# 余白を調整
plt.subplots_adjust(wspace=0.3)  # 画像間の余白

# 画像を保存
plt.savefig('output_image_with_margin_and_axes_and_viridis_cmap.png', bbox_inches='tight', pad_inches=0.1)

# 表示
plt.show()
