import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


class ResNet18Embedder:
    def __init__(self, device: str = 'cpu'):
        # 使用デバイス（CPU or GPU）の指定
        self.device = device

        # 1. 事前学習済みのResNet18を読み込む
        self.model = models.resnet18(pretrained=True)

        # 2. 分類層を恒等写像に置き換え、特徴ベクトルのみ取得できるようにする
        self.model.fc = nn.Identity()

        # モデルを指定デバイスに転送し、推論モードに
        self.model.to(self.device)
        self.model.eval()

        # 3. 前処理の定義
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ResNetの入力サイズに合わせる
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNetの平均
                std=[0.229, 0.224, 0.225]    # ImageNetの標準偏差
            )
        ])

    def __call__(self,
                 image_path: str = "",
                 img=None,
                 ) -> torch.Tensor:
        """画像パスを入力して、正規化した特徴ベクトルを返す"""
        if img is None:
            img = Image.open(image_path)
        # 4. 推論(特徴ベクトルの取得)
        with torch.no_grad():
            # 前処理とバッチ次元の追加
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)

            # ResNetから埋め込みベクトルを取得
            embedding = self.model(img_tensor)  # shape: [1, 512] (ResNet18の場合)

            # 特徴ベクトルの正規化（オプション）
            embedding_norm = embedding / embedding.norm(dim=1, keepdim=True)

        # バッチ次元を落として返す (shape: [512])
        return embedding_norm.squeeze(0)
