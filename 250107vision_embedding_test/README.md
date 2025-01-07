# 環境構築
- 使用マシン: A100
  - NVIDIA-SMI 550.120
  - Driver Version: 550.120
  - CUDA Version: 12.4 

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

~~~