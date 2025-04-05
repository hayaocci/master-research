# Title: Unreal Engine 5 でデータセットを自動生成する


import unreal
import os

# 保存先のパスを定義
save_path = "C:/workspace"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 保存するファイル名
image_filename = os.path.join(save_path, "captured_image.png")

# アクター取得（EditorActorSubsystemを使用）
editor_actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
all_actors = editor_actor_subsystem.get_all_level_actors()

# CameraActor を探す（名前に "CameraActor" が含まれるものを見つける）
camera_actor = None
for actor in all_actors:
    actor_name = actor.get_name()
    unreal.log("Found actor: " + actor_name)
    if "CameraActor" in actor_name:
        camera_actor = actor
        unreal.log("→ CameraActor として使用: " + actor_name)
        break

if not camera_actor:
    unreal.log_error("CameraActor が見つかりません。名前が正しいか確認してください。")
else:
    # Viewport のカメラを CameraActor の位置にセット
    loc = camera_actor.get_actor_location()
    rot = camera_actor.get_actor_rotation()
    unreal.EditorLevelLibrary.set_level_viewport_camera_info(loc, rot)

    # スクリーンショットを撮影
    unreal.AutomationLibrary.take_high_res_screenshot(
        1920, 1080, image_filename
    )

    unreal.log("スクリーンショットが保存されました: " + image_filename)
