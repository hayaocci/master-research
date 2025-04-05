# Blender実行ファイルのパス
$blenderExe = "C:\Program Files\Blender Foundation\Blender 3.6\blender.exe"

# .blendファイルのパス
$blendScript = "C:/workspace/Github/master-research/blender3/h2a_1226_dataset_maker_joined.blend"

# Pythonスクリプトのパス
$pythonScript = "C:\workspace\Github\master-research\blender3\blender_python\dataset_maker.py"

# 中断を検知するイベント登録
$eventHandler = {
    Write-Host "`n中断されました。Blenderプロセスを終了します..."
    Stop-Process -Name "blender" -Force -ErrorAction SilentlyContinue
    exit
}
Register-EngineEvent PowerShell.Exiting -Action $eventHandler | Out-Null
$cancelled = $false

# Ctrl+C 用のハンドラーも設定
[Console]::TreatControlCAsInput = $false
$null = Register-ObjectEvent -InputObject $Host -EventName "CancelKeyPress" -Action {
    Write-Host "`nCtrl+C を検出しました。終了処理を行います..."
    $global:cancelled = $true
}

# メインループ
for ($i = 1; $i -le 50; $i++) {
    if ($cancelled) { break }

    Write-Host "実行回数: $i"
    try {
        & "$blenderExe" "$blendScript" --python "$pythonScript"
    } catch {
        Write-Host "エラーが発生しました: $_"
    } finally {
        try {
            Stop-Process -Name "blender" -Force -ErrorAction SilentlyContinue
        } catch { }
    }
}

Write-Host "スクリプトを終了しました。"


# # Blender実行ファイルのパス（環境に合わせて変更してください）
# $blenderExe = "C:\Program Files\Blender Foundation\Blender 3.6\blender.exe"

# # Blenderスクリプトのパス（環境に合わせて変更してください）
# $blendScript = "C:/workspace/Github/master-research/blender3/h2a_1226_dataset_maker_joined.blend"


# # Pythonスクリプト(.py)のパス（レンダリング処理を実装したスクリプトのパス）
# $pythonScript = "C:\workspace\Github\master-research\blender3\blender_python\dataset_maker.py"

# # 16回繰り返す
# for ($i = 1; $i -le 50; $i++) {
#     Write-Host "実行回数: $i"
#     try {
#         & "$blenderExe" "$blendScript" --python "$pythonScript"
#     } catch {
#         Write-Host "エラーが発生しました: $_"
#     } finally {
#         # 必要であればBlenderのプロセスを強制終了
#         Stop-Process -Name "blender" -Force
#     }
# }

# Write-Host "すべての実行が完了しました。"
