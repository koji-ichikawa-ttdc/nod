# Nearby Object Detection

YOLOXを使用して物体検出した後にBBOXの幅から近接物体を検出する

##環境準備
環境はVSCode + Docker + Dev Container
* VSCodeをインストール
* VSCode拡張機能 Dockerをインストール
* VSCode拡張機能 Dev Containersをインストール
* VSCodeからReopen in Container(コンテナーで再度開く)でコンテナに入る

## YOLOX準備
```sh
git clone https://github.com/Megvii-BaseDetection/YOLOX
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth
pip install -v -e /workspaces/nod/YOLOX
```
今回はModelはyolox_nano.pthを使用  
他の[Model](https://github.com/Megvii-BaseDetection/YOLOX)も使用可能

## 動画準備
```sh
mkdir /workspaces/nod/data
```
dataフォルダに動画を入れる

## 実行
```sh
python nearby_od.py video -n yolox-nano -c yolox_nano.pth --path data/sample.AVI --save_result --object_width 300
```
* --path 動画ファイルのパスを入力
* --object_width 検出するBBOXの幅を入力

## 実行結果
* --object_width以上のフレームだけYOLO_outputに動画出力される
* 動画の左上に検出した時間とフレーム番号が記載してある
* 動画と同じフォルダに全BBOXのjsonが保存される。検出がない場合はjson出力されない。