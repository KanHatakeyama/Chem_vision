# 概要
- qwen vlでアノテーションする

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
pip install qwen-vl-utils==0.0.8
#pip install transformers==4.5.1 # rust error
pip install transformers==4.47.1
pip install accelerate==0.34.2
~~~