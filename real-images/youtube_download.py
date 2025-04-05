# from pytube import YouTube

# channel_url = 'https://www.youtube.com/watch?v=W7dtphs105E&list=TLGG7Y74wWvb_u0wNDA0MjAyNQ'
# channel_url = 'https://www.youtube.com/watch?v=7cxXP9OxQAc&list=TLGGED3oX-lSPtUwNTA0MjAyNQ'
# channel_url = 'https://www.youtube.com/watch?v=XdwyxPqy6OI&list=TLGGuHDPmXfyMKAwNTA0MjAyNQ'
# channel_url = 'https://www.youtube.com/watch?v=QUvHq2g3viw&list=TLGGEQ9zFj3XaLowNTA0MjAyNQ'
channel_url = 'https://www.youtube.com/watch?v=N71w4Mgvmcg&list=TLGGMnUpOHKF0p8wNTA0MjAyNQ'

# yt = YouTube(url)

# # 音声を含むストリームを取得（通常は最高解像度）
# stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()

# # 動画のダウンロード（現在のディレクトリに保存）
# stream.download()

import yt_dlp as youtube_dl

def download_channel_videos(channel_url):
    # ダウンロードオプションを定義
    ydl_opts = {
        'format': 'best',  # 動画の最適なフォーマットをダウンロード
        'outtmpl': '%(title)s.%(ext)s',  # 保存するファイル名のフォーマット
        'ignoreerrors': True  # エラーがあっても処理を継続する
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([channel_url])

# チャンネルのURLを指定
# channel_url = 'https://www.youtube.com/@/チャンネルのID/videos'
download_channel_videos(channel_url)

