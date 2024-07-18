import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_histogram(image):
    hist = []
    for i in range(3):
        hist.append(cv2.calcHist([image], [i], None, [256], [0, 256]))
        plt.plot(hist[i], color = ['b', 'g', 'r'][i])
        plt.xlim([0, 256])
    plt.show()
    return hist

def compare_histograms(hist1, hist2):
    comparison = []
    for i in range(3):
        comparison.append(cv2.compareHist(hist1[i], hist2[i], cv2.HISTCMP_CORREL))
    return comparison

def plot_histograms(hist1, hist2):
    colors = ('b', 'g', 'r')
    for i, color in enumerate(colors):
        plt.subplot(2, 3, i+1)
        plt.title(f'Image 1 - {color.upper()} Channel')
        plt.plot(hist1[i], color=color)
        plt.xlim([0, 256])
        
        plt.subplot(2, 3, i+4)
        plt.title(f'Image 2 - {color.upper()} Channel')
        plt.plot(hist2[i], color=color)
        plt.xlim([0, 256])
    
    plt.tight_layout()
    plt.show()

def gaussian_blur(image, kernel_size=21):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 21)



# 画像を読み込む
# img1 = cv2.imread('../data/real_img/0.png')
# img2 = cv2.imread('../data/real_img/1.jpg')

img1 = cv2.imread('8.png')
img2 = cv2.imread('7.jpg')



cv2.imshow('image', gaussian_blur(img1))
cv2.imwrite('8_blur.png', gaussian_blur(img1))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('image', gaussian_blur(img2))
cv2.waitKey(0)
cv2.destroyAllWindows()



# BGR to RGB
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# ヒストグラムを計算
hist1 = calculate_histogram(img1)
hist2 = calculate_histogram(img2)

# ヒストグラムを比較
comparison = compare_histograms(hist1, hist2)

# 結果を表示
print("Histogram Comparison (Correlation):")
print(f"R Channel: {comparison[0]:.4f}")
print(f"G Channel: {comparison[1]:.4f}")
print(f"B Channel: {comparison[2]:.4f}")

# ヒストグラムをプロット
plot_histograms(hist1, hist2)
