# 概要
- resnet18で画像をembedする
- 動画から画像を抽出し､レアイベントを抽出する


# setup
~~~
conda create -n vision python=3.10 -y  
conda activate vision


#cuda 12.4の場合
#pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

#cuda 11.8
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118

pip install scikit-learn
pip install seaborn
pip install umap-learn

#1/8
pip install opencv-python==4.10.0.84

~~~