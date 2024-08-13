# モジュール関数
import os


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

# def save_file_in_series(save_dir_path)
