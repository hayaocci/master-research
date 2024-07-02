# # blender path
# $blenderPath = "C:\Program FIles\Blender Foundation\Blender 4.1\blender.exe"

# # blender file path
# $blenderFilePath = "C:\workspace\MasterResearch\blender\new_earth_ver1.03_scripting_withCamera\new_earth\earth_debris_scripting_withCamera.blend"


# # python file path
# $pythonScriptPath = "C:\workspace\Github\master-research\blender\make_dataset.py"

# & "$blenderPath" "$blendFilePath" --python "$pythonScriptPath"

# blender path
$blenderPath = "C:\Program Files\Blender Foundation\Blender 4.1\blender.exe"

# blender file path
$blendFilePath = "C:\workspace\MasterResearch\blender\new_earth_ver1.03_scripting_withCamera\new_earth\earth_debris_scripting_withCamera.blend"

# python file path
$pythonScriptPath = "C:\workspace\Github\master-research\blender\make_dataset.py"

# スクリプトの実行
try {
    & "$blenderPath" "$blendFilePath" --python "$pythonScriptPath"
} catch {
    Write-Host "エラーが発生しました: $_"
}


# To run this script, run the following command in the terminal
# PowerShell.exe -ExecutionPolicy Bypass -File ".\make_dataset_withBlender.ps1"