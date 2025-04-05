import os

def get_last_number_in_directory(dir_path: str) -> int:
    """
    指定したディレクトリ直下にある、数字のみの名前からなるサブディレクトリのうち
    最大の数値を返します。該当するサブディレクトリがない場合は None を返します。
    """
    candidates = []
    for item in os.listdir(dir_path):
        full_path = os.path.join(dir_path, item)
        # ディレクトリかつ数字の名前であるかを判定
        if os.path.isdir(full_path) and item.isdigit():
            candidates.append(int(item))

    return max(candidates) if candidates else None