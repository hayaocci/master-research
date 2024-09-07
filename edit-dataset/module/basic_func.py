# モジュール関数
import os
import cv2
import numpy as np

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

def calculate_centroid(image_path):
    # 画像をグレースケールで読み込み
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 画像が正しく読み込まれたか確認
    if img is None:
        print("画像を読み込めませんでした。")
        return None, None
    
    # 画像が既に2値化されていない場合は、ここで2値化を行う
    _, binary_img = cv2.threshold(img, 5, 255, cv2.THRESH_BINARY)
    
    # モーメントを計算
    moments = cv2.moments(img)
    
    # 重心を計算
    if moments["m00"] != 0:
        cX = int(moments["m10"] / moments["m00"])
        cY = int(moments["m01"] / moments["m00"])
    else:
        print("重心を計算できませんでした。")
        return None, None
    
    return cX, cY

def calculate_bbox(image_path):
    # 画像をグレースケールで読み込み
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 画像が正しく読み込まれたか確認
    if img is None:
        print("画像を読み込めませんでした。")
        return None, None, None, None
    
    # 画像を2値化（しきい値は適宜調整）
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # 白色部分の輪郭を検出
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 最大の輪郭を選択
    if contours:
        # 輪郭のバウンディングボックスを取得
        x, y, w, h = cv2.boundingRect(contours[0])
        
        # バウンディングボックスの中心座標を計算
        cx = x + w // 2
        cy = y + h // 2
        
        return (cx, cy), w, h
    else:
        print("白色部分が検出されませんでした。")
        return None, None, None, None
    
def calculate_bbox_and_draw(image_path):
    # 画像をグレースケールで読み込み
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 画像が正しく読み込まれたか確認
    if img is None:
        print("画像を読み込めませんでした。")
        return None, None, None, None, None, None, None
    
    # 画像をカラーに変換（BBOXを描画するため）
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # 画像を2値化（しきい値は適宜調整）
    _, binary_img = cv2.threshold(img, 5, 255, cv2.THRESH_BINARY)
    
    # 白色部分の輪郭を検出
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 最大の輪郭を選択
    if contours:
        # 輪郭のバウンディングボックスを取得
        x, y, w, h = cv2.boundingRect(contours[0])
        
        # バウンディングボックスの中心座標を計算
        cx = x + w // 2
        cy = y + h // 2
        
        # 画像の幅と高さを取得
        img_height, img_width = img.shape
        
        # 正規化された中心座標と幅、高さを計算
        norm_cx = cx / img_width
        norm_cy = cy / img_height
        norm_w = w / img_width
        norm_h = h / img_height
        
        # BBOXを描画（青色、太さ2）
        cv2.rectangle(img_color, (x, y), (x + w, y + h), (255, 0, 0), 1)
        
        # 結果の画像を保存
        # cv2.imwrite(output_path, img_color)
        
        # return (cx, cy), (norm_cx, norm_cy), w, h, norm_w, norm_h, img_color
        # return (cx, cy, w, h), (norm_cx, norm_cy, norm_w, norm_h)
        return cx, cy, w, h, norm_cx, norm_cy, norm_w, norm_h
    else:
        print("白色部分が検出されませんでした。")
        return None, None, None, None, None, None, None
    
def calculate_bbox_and_draw2(image_path):
    """
    bboxの左上と右下の座標を返す
    """

    # 画像をグレースケールで読み込み
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 画像が正しく読み込まれたか確認
    if img is None:
        print("画像を読み込めませんでした。")
        return None, None, None, None, None, None, None
    
    # 画像をカラーに変換（BBOXを描画するため）
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # 画像を2値化（しきい値は適宜調整）
    _, binary_img = cv2.threshold(img, 5, 255, cv2.THRESH_BINARY)
    
    # 白色部分の輪郭を検出
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 最大の輪郭を選択
    if contours:
        # 輪郭のバウンディングボックスを取得
        x1, y1, w, h = cv2.boundingRect(contours[0])
        
        # バウンディングボックスの右下座標を計算
        x2 = x1 + w
        y2 = y1 + h
        
        # 画像の幅と高さを取得
        img_height, img_width = img.shape
        
        # 正規化された中心座標と幅、高さを計算
        norm_x1 = x1 / img_width
        norm_y1 = y1 / img_height
        norm_x2 = x2 / img_width
        norm_y2 = y2 / img_height
        
        # BBOXを描画（青色、太さ2）
        cv2.rectangle(img_color, (x1, y1), (x2, y2), (255, 0, 0), 1)

        # 結果の画像を保存
        # cv2.imwrite(output_path, img_color)
        
        # return (cx, cy), (norm_cx, norm_cy), w, h, norm_w, norm_h, img_color
        # return (cx, cy, w, h), (norm_cx, norm_cy, norm_w, norm_h)
        return x1, y1, x2, y2, norm_x1, norm_y1, norm_x2, norm_y2, img_height, img_width
    else:
        print("白色部分が検出されませんでした。")
        return None, None, None, None, None, None, None, None, None, None

def extract_and_save_bbox(image_path, output_path):
    # 画像をグレースケールで読み込み
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 画像が正しく読み込まれたか確認
    if img is None:
        print("画像を読み込めませんでした。")
        return None, None, None, None, None, None, None
    
    # 画像を2値化（しきい値は適宜調整）
    _, binary_img = cv2.threshold(img, 5, 255, cv2.THRESH_BINARY)
    
    # 白色部分の輪郭を検出
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 最大の輪郭を選択
    if contours:
        # 輪郭のバウンディングボックスを取得
        x, y, w, h = cv2.boundingRect(contours[0])
        
        # バウンディングボックス内の画像を切り出し
        cropped_img = img[y:y+h, x:x+w]
        
        # 画像の幅と高さを取得
        img_height, img_width = img.shape
        
        # 正規化された中心座標と幅、高さを計算
        cx = x + w // 2
        cy = y + h // 2
        norm_cx = cx / img_width
        norm_cy = cy / img_height
        norm_w = w / img_width
        norm_h = h / img_height
        
        # 切り出した画像を保存
        cv2.imwrite(output_path, cropped_img)
        
        return (cx, cy), (norm_cx, norm_cy), w, h, norm_w, norm_h, cropped_img
    else:
        print("白色部分が検出されませんでした。")
        return None, None, None, None, None, None, None
    
def resize_to_square(image_path, output_path):
    # 画像を読み込み
    img = cv2.imread(image_path)
    
    # 画像が正しく読み込まれたか確認
    if img is None:
        print("画像を読み込めませんでした。")
        return None
    
    # 画像の高さと幅を取得
    height, width = img.shape[:2]
    
    # 正方形の一辺の長さは短辺に合わせる
    side_length = min(height, width)
    
    # 画像を正方形にリサイズ
    resized_img = cv2.resize(img, (side_length, side_length), interpolation=cv2.INTER_AREA)
    
    # 結果の画像を保存
    cv2.imwrite(output_path, resized_img)
    
    return resized_img

def make_seg_label(image_path, output_path=None):
    import numpy as np

    # 画像をグレースケールで読み込み
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
    # 画像が正しく読み込まれたか確認
    if img is None:
        print(f"{image_path} を読み込めませんでした。")
        
    # 画像を2値化（しきい値は適宜調整）
    _, binary_img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)

    # 12x12にリサイズ
    output_img = cv2.resize(binary_img, (12, 12), interpolation=cv2.INTER_AREA)

    # 2値化
    _, output_img = cv2.threshold(output_img, 1, 255, cv2.THRESH_BINARY)

    # 出力画像を保存
    # cv2.imwrite(output_path, output_img)

    return output_img



def combine_imgs(large_img, small_img):
    # 画像の高さと幅を取得
    large_height, large_width = large_img.shape[:2]
    small_height, small_width = small_img.shape[:2]

    # 大きい画像のサイズに合わせて小さい画像をリサイズ
    resized_small_img = cv2.resize(small_img, (large_width, large_height), interpolation=cv2.INTER_AREA)

    # リサイズした画像を上に重ねる
    combined_img = cv2.addWeighted(large_img, 1.0, resized_small_img, 0.5, 0)

    # 画像を保存
    cv2.imwrite("combined_image.png", combined_img)


if __name__ == "__main__":

    image_path = "bin_sample.png"
    # output_path = "bbox_sample.png"
    output_path = 'seg_label.png'

    make_seg_label(image_path, output_path)

    combine_imgs(cv2.imread(image_path), cv2.imread(output_path))

    # 使用例
    # bbox_center, width, height = calculate_bbox_and_draw(image_path)
    # if bbox_center is not None:
    #     print(f"BBOXの中心座標: {bbox_center}")
    #     print(f"BBOXの幅: {width}")
    #     print(f"BBOXの高さ: {height}")

    # bbox_center, norm_bbox_center, width, height, norm_width, norm_height, output_img = calculate_bbox_and_draw(image_path, output_path)
    # if bbox_center is not None:
    #     print(f"BBOXの中心座標: {bbox_center}")
    #     print(f"正規化されたBBOXの中心座標: {norm_bbox_center}")
    #     print(f"BBOXの幅: {width}")
    #     print(f"正規化されたBBOXの幅: {norm_width}")
    #     print(f"BBOXの高さ: {height}")
    #     print(f"正規化されたBBOXの高さ: {norm_height}")
    #     print(f"BBOXを描画した画像が 'output_image.png' として保存されました。")

    # # 使用例
    # bbox_center, norm_bbox_center, width, height, norm_width, norm_height, cropped_img = extract_and_save_bbox(image_path, output_path)
    # if bbox_center is not None:
    #     print(f"BBOXの中心座標: {bbox_center}")
    #     print(f"正規化されたBBOXの中心座標: {norm_bbox_center}")
    #     print(f"BBOXの幅: {width}")
    #     print(f"正規化されたBBOXの幅: {norm_width}")
    #     print(f"BBOXの高さ: {height}")
    #     print(f"正規化されたBBOXの高さ: {norm_height}")
    #     print(f"切り出した画像が 'cropped_image.png' として保存されました。")

    # # 使用例
    # resized_img = resize_to_square("bbox_sample.png", "output_resized_square_image.png")
    # if resized_img is not None:
    #     print("画像を圧縮して正方形にリサイズし、'output_resized_square_image.png' として保存しました。")