# master-research

### データセット作成
Blender内でPythonファイルを実行する -> 途中で処理落ちしてしまう  
この問題を解消するためにPowershellを用いて実行することにした  

### 実行環境
py391


#### 2024/06/20
Blenderを用いて地球のモデルを作成
-> アルベド光の再現が必要  
参考になりそうなURL  
https://design.kyusan-u.ac.jp/OpenSquareJP/?Blender/Rendering_Lights#:~:text=%E2%86%91-,%E5%85%89%E3%81%AE%E5%BC%B7%E3%81%95,%E3%82%92%E5%8F%82%E8%80%83%E3%81%AB%E3%81%97%E3%81%A6%E3%81%8F%E3%81%A0%E3%81%95%E3%81%84%E3%80%82  
-> 大気の厚さはあまり再現しなくてもいいかも？？  

#### 2024/06/21  
PAFの鏡面反射の再現  
-> Shade Smoothを用いて行うも、画像のように縞模様が入ってしまう  

#### 2024/06/24
Blenderのレンダリング中のバグの解決に役に立ちそうなIssueなど  
https://projects.blender.org/blender/blender/issues/117229

#### 2024/07/02
Blenderでデータセットを作成する際にGPUのメモリに問題があるため処理が落ちてしまう問題  
> Powershellで実行するプログラムを作成した

#### 2024/07/09
master-research ディレクトリの整理

#### 2025/04/05
real-images added
