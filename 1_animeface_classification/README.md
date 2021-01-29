1_classification_animeface
====
アニメのキャラクターの顔を集めたデータセットを用いた識別

## Description
### 使用データセット
[AnimeFace Character Dataset](http://www.nurs.or.jp/%7Enagadomi/animeface-character-dataset/README.html)

### 使用ネットワーク
[Mobilenet v2](https://arxiv.org/abs/1801.04381) [1]

### 参考文献
[1] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen, "MobileNetV2: Inverted Residuals and Linear Bottlenecks," CVPR, 2018.  

## Usage
### 実行
```
# ダウンロード，フォルダ構成
python py/setup.py
# 学習，識別
bash train.sh
# ログ確認
tensorboard --logdir=log/animeface
```

## Result
Top1 Accuracy 92.03%  
elapsed time = 0h 8m 14s
