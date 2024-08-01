import cv2
import numpy as np

def calculate_centroid(image_path):
    # 画像をグレースケールで読み込み
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 画像が正しく読み込まれたか確認
    if img is None:
        print("画像を読み込めませんでした。")
        return None, None
    
    # 画像が既に2値化されていない場合は、ここで2値化を行う
    # _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # モーメントを計算
    moments = cv2.moments(img)
    
    # 重心を計算
    if moments["m00"] != 0:
        cX = int(moments["m10"] / moments["m00"])
        cY = int(moments["m01"] / moments["m00"])
    else:
        print("重心を計算できませんでした。")
        return None, None
    
    return (cX, cY), img

def draw_centroid(image_path, centroid):
    # 元の画像をカラーで読み込み
    original_img = cv2.imread(image_path)
    
    if original_img is None:
        print("元の画像を読み込めませんでした。")
        return None
    
    # 重心位置に赤い点を描画
    # cv2.circle(original_img, centroid, 5, (0, 0, 255), -1)
    # 重心位置に3×3の赤い点を描画
    cv2.rectangle(original_img, 
                (centroid[0]-1, centroid[1]-1), 
                (centroid[0]+1, centroid[1]+1), 
                (0, 0, 255), 
                -1)
    
    # 重心の座標を画像に表示
    # cv2.putText(original_img, f"({centroid[0]}, {centroid[1]})", 
    #             (centroid[0] + 10, centroid[1]), 
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 2)
    
    return original_img

# 画像ファイルのパス
# image_path = "path_to_your_image.png"
image_path = 'label_example.png'

# 重心を計算
centroid, _ = calculate_centroid(image_path)

if centroid:
    print(f"画像の重心位置: X = {centroid[0]}, Y = {centroid[1]}")
    
    # 重心を描画した画像を取得
    result_img = draw_centroid(image_path, centroid)
    
    if result_img is not None:
        # 結果を表示
        cv2.imshow("Centroid", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 結果を保存（オプション）
        cv2.imwrite("result_with_centroid.png", result_img)
        print("結果を 'result_with_centroid.png' として保存しました。")
else:
    print("重心の計算に失敗しました。")