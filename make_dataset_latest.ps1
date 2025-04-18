﻿# @echo off

# # blender path
# $blenderPath = "C:\Program Files\Blender Foundation\Blender 4.1\blender.exe"

# # blender file path
# $blendFilePath = "C:\workspace\MasterResearch\blender\new_earth_ver1.03_scripting_withCamera\new_earth\earth_debris_scripting_withCamera.blend"

# # python file path
# $pythonScriptPath = "C:\workspace\Github\master-research\blender\make_dataset.py"

# # スクリプトの実行
# try {
#     & "$blenderPath" "$blendFilePath" --python "$pythonScriptPath"
# } catch {
#     Write-Host "エラーが発生しました: $_"
# }

# $echo off

# blender path
$blenderPath = "C:\Program Files\Blender Foundation\Blender 4.1\blender.exe"

# blender file path
# $blendFilePath = "C:\workspace\MasterResearch\blender\new_earth_ver1.10_scripting_withCamera\new_earth\earth_debris_scripting_withCamera.blend"
$blendFilePath = "C:\workspace\MasterResearch\blender\new_earth_ver1.10_scripting_withCamera_sun_synchronous\new_earth\sun_synchronous_orbit_1.10.blend"
# python file path
# $pythonScriptPath = "C:\workspace\Github\master-research\blender\make_dataset.py" # 古いバージョン
$pythonScriptPath = "C:\workspace\Github\master-research\blender\make_dataset_withPowerShell_ver4.py"

# 16回繰り返す
for ($i = 1; $i -le 100; $i++) {
    Write-Host "実行回数: $i"
    try {
        & "$blenderPath" "$blendFilePath" --python "$pythonScriptPath"
    } catch {
        Write-Host "エラーが発生しました: $_"
    } finally {
        # 必要であればBlenderのプロセスを強制終了
        Stop-Process -Name "blender" -Force
    }
}

Write-Host "すべての実行が完了しました。"

# To run this script, run the following command in the terminal
# PowerShell.exe -ExecutionPolicy Bypass -File ".\make_dataset_latest.ps1"