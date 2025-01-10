#!/usr/bin/env python
# coding: utf-8
"""
使用例：
python extract_frames.py \
  --video_path movie/20241126ikura_full.mp4 \
  --out_dir out_images \
  --sim_threshold 0.85 \
  --device cpu


"""


import argparse
import os
import cv2
from tqdm import tqdm
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from Resnet import ResNet18Embedder


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from a video based on similarity threshold, checking against existing images in out_dir. \
                     Saved image files are named by <video_basename>_<frame_number>.jpg"
    )

    parser.add_argument("--video_path", type=str, required=True,
                        help="Path to the input video file (e.g., movie/20241126ikura_full.mp4)")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory where extracted images are stored/loaded")
    parser.add_argument("--sim_threshold", type=float, default=0.85,
                        help="Similarity threshold to decide if a frame is 'unique'")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to run ResNet18Embedder on (e.g., 'cpu' or 'cuda')")

    args = parser.parse_args()

    video_path = args.video_path
    out_dir = args.out_dir
    sim_threshold = args.sim_threshold
    device = args.device

    # Embeddingモデルの初期化
    embedder = ResNet18Embedder(device=device)

    # 出力先のディレクトリがなければ作成
    os.makedirs(out_dir, exist_ok=True)

    # 1) 既存画像ファイルを読み込み、Embeddingをリストに格納
    existing_files = os.listdir(out_dir)
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp")
    embed_list = []

    for filename in existing_files:
        # 画像拡張子以外はスキップ
        if not filename.lower().endswith(valid_exts):
            continue
        path = os.path.join(out_dir, filename)

        # 画像を読み込み -> PIL形式に変換 -> Embedding
        existing_img_bgr = cv2.imread(path)
        if existing_img_bgr is None:
            # 読み込めない画像はスキップ
            continue

        existing_img_rgb = cv2.cvtColor(existing_img_bgr, cv2.COLOR_BGR2RGB)
        existing_img_pil = Image.fromarray(existing_img_rgb)

        # Embeddingを取得
        embed = embedder(img=existing_img_pil)
        embed_list.append(embed)

    # 2) 動画ファイル名のベースを取得 (拡張子を除いたもの)
    #   例: "movie/20241126ikura_full.mp4" -> "20241126ikura_full"
    video_basename = os.path.splitext(os.path.basename(video_path))[0]

    # 動画読み込み開始
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:  # 万が一fpsが取得できない場合の対処
        fps = 30.0

    # 動画のフレームを順次処理
    pbar = tqdm(total=total_frames, desc="Processing frames")

    added_count = 0  # 新規保存したフレーム数をカウント

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 現在のフレーム番号（0始まり）
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1  #
        # あるいは独自カウンタで回すなら
        # frame_idx = pbar.n

        # PIL形式に変換してEmbeddingを抽出
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        embed = embedder(img=img_pil)

        # 既存 Embedding (embed_list) との類似度をチェック
        """
        sim_list = []
        for old_embed in embed_list:
            sim = cosine_similarity(
                embed.reshape(1, -1),
                old_embed.reshape(1, -1)
            )[0][0]  # cosine_similarityは [[値]] の2次元で返す
            sim_list.append(sim)
        """
        embed_list_np = np.array(embed_list)  # リストなら先に NumPy 配列に変換する
        sim_list = cosine_similarity(
            embed.reshape(1, -1),  # (1, d)
            embed_list_np          # (N, d)
        )[0]                       # 結果は (1, N) なので [0] で (N,) にする

        # embed_listが空の場合は類似度0とみなす
        if len(sim_list) == 0:
            min_sim = 0.0
        else:
            min_sim = max(sim_list)

        # 閾値より低い -> 新しいフレームとして保存
        if min_sim < sim_threshold:
            # 「動画ファイル名_フレーム番号.jpg」で保存
            out_filename = f"{video_basename}_{frame_idx}.jpg"
            out_path = os.path.join(out_dir, out_filename)
            cv2.imwrite(out_path, frame)

            # embed_list に追加
            embed_list.append(embed)

            print("new frame added: ", frame_idx)
            added_count += 1

        pbar.update(1)

    pbar.close()
    cap.release()
    cv2.destroyAllWindows()

    print(
        f"Total unique images (existing + newly extracted): {len(embed_list)}")
    print(f"Newly added frames from video: {added_count}")
    print(f"Saved frames to: {out_dir}")


if __name__ == "__main__":
    main()
