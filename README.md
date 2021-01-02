tensorflow(2.x) tutorial for japanese
====

## Description
[1_animeface_classification](https://github.com/chatflip/tutorial_tensorflow_japanese/tree/master/1_animeface_classification)  
フォルダ分けしたデータセットの画像識別  

## Installation
### anaconda導入
最新バージョンを使う場合[ここ](https://www.anaconda.com/distribution/)からダウンロード, インストール  
実行時の環境は[ここ](https://repo.continuum.io/archive/) から```Anaconda3-5.2.0-MacOSX-x86_64.sh``` をダウンロード, インストール

### pytorch導入(仮想環境)
``` 
conda create -n tf24 python=3.7 -y  
source activate tf24  
pip install tensorflow-gpu==2.4.0 scipy pillow opencv-python albumentations
```
