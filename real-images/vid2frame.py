import cv2
import os

def save_all_frames(video_path, dir_path, basename, ext='png'):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    n = 0

    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imwrite('{}_{}.{}'.format(base_path, str(n).zfill(digit), ext), frame)
            n += 1
        else:
            return

import cv2
import os
import numpy as np

def save_changed_frames(video_path, dir_path, basename, ext='png', threshold=0.1):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Failed to open video.")
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)
    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    prev_gray = None
    n = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            # 差分を計算し、平均の絶対差を算出
            diff = cv2.absdiff(gray, prev_gray)
            mean_diff = np.mean(diff)

            if mean_diff < threshold:
                n += 1
                continue

        # 変化があったとみなして保存
        filename = f"{base_path}_{str(n).zfill(digit)}.{ext}"
        cv2.imwrite(filename, frame)
        saved_count += 1

        prev_gray = gray
        n += 1

    print(f"{saved_count} frames saved.")



if __name__ == '__main__':

    # video_path = 'crd2_adrasj_still.mp4'
    # video_path = 'crd2_adrasj_fly-around-observation-wide.mp4'
    video_path = 'crd2_adrasj_fly-around-observation-tele_0716.mp4'
    # dir_path = 'frames/crd2_adrasj_still'
    # dir_path = 'frames/crd2_adrasj_fly-around-observation-wide'
    dir_path = 'frames/crd2_adrasj_fly-around-observation-tele_0716'
    basename = ''
    # save_all_frames(video_path, dir_path, basename)
    save_changed_frames(video_path, dir_path, basename, threshold=1.0)
    print('done')