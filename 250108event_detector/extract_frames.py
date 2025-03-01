#!/usr/bin/env python
# coding: utf-8
"""
使用例：
python extract_frames.py \
  --video_dir /data/2024/Chem_vision/private_data/movies \
  --out_dir out_images \
  --sim_threshold 0.8 \
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
        description="指定フォルダ内のすべての動画について、類似度しきい値をもとにフレームを抽出するスクリプト。\n"
                    "既存の画像が out_dir に存在する場合、それらのEmbeddingを使って、重複(類似)判定を行います。"
    )

    parser.add_argument("--video_dir", type=str, required=True,
                        help="動画ファイルが複数入ったフォルダへのパス (例: movie)")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="抽出した画像を保存するフォルダパス (既存画像もここから読み込まれます)")
    parser.add_argument("--sim_threshold", type=float, default=0.85,
                        help="フレームの類似度がこの値以上ならスキップ、それより低ければ新規保存する。")
    parser.add_argument("--device", type=str, default="cpu",
                        help="ResNet18Embedder を実行するデバイス ('cpu' あるいは 'cuda')")

    args = parser.parse_args()

    video_dir = args.video_dir
    out_dir = args.out_dir
    sim_threshold = args.sim_threshold
    device = args.device

    # ----- Embeddingモデルの初期化 -----
    embedder = ResNet18Embedder(device=device)

    # ----- 出力先フォルダの作成 -----
    os.makedirs(out_dir, exist_ok=True)

    # ----- 既存画像ファイルの Embedding をリストに格納 -----
    existing_files = os.listdir(out_dir)
    valid_img_exts = (".png", ".jpg", ".jpeg", ".bmp")
    embed_list = []

    print("[INFO] Loading existing images' embeddings...")
    for filename in existing_files:
        if not filename.lower().endswith(valid_img_exts):
            continue
        path = os.path.join(out_dir, filename)

        # 画像を読み込み -> PILに変換 -> Embedding
        existing_img_bgr = cv2.imread(path)
        if existing_img_bgr is None:
            # 読み込めない画像はスキップ
            continue

        existing_img_rgb = cv2.cvtColor(existing_img_bgr, cv2.COLOR_BGR2RGB)
        existing_img_pil = Image.fromarray(existing_img_rgb)

        # Embedding
        embed = embedder(img=existing_img_pil)
        embed_list.append(embed)

    print(
        f"[INFO] Loaded {len(embed_list)} embeddings from existing images.\n")

    # ----- 指定フォルダ内の動画ファイルを一括処理 -----
    valid_video_exts = (".mp4", ".avi", ".mov", ".mkv")  # 必要に応じて拡張子を追加

    video_files = []
    for root, dirs, files in os.walk(video_dir):
        for f in files:
            if f.lower().endswith(valid_video_exts):
                video_files.append(os.path.join(root, f))
        if not video_files:
            print(
                f"[WARN] 指定フォルダ({video_dir})に該当する動画ファイルが見つかりませんでした。処理を終了します。")
            return

    print(
        f"[INFO] Found {len(video_files)} video file(s) in '{video_dir}'. Processing...\n")

    # 総合カウンタ（すべての動画を通して新規保存したフレーム数）
    total_added_count = 0

    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)

        # 動画名の拡張子を除いた部分
        video_basename = os.path.splitext(os.path.basename(video_path))[0]

        # 動画を開く
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {video_path}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:  # fpsが取得できない場合の対処
            fps = 30.0

        print(
            f"[INFO] Processing '{video_file}' (Frames: {total_frames}, FPS: {fps})")

        pbar = tqdm(total=total_frames, desc=f"Extracting from {video_file}")
        added_count = 0  # この動画ファイルに対して保存したフレーム数

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 現在のフレーム番号
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

            # Embedding取得
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            embed = embedder(img=img_pil)

            # embed_list が空の場合は、問答無用で新規保存
            if len(embed_list) == 0:
                min_sim = 0.0
            else:
                # 既存Embeddingsとの類似度を計算
                embed_list_np = np.array(embed_list)  # (N, d)
                sim_list = cosine_similarity(
                    embed.reshape(1, -1),  # (1, d)
                    embed_list_np          # (N, d)
                )[0]
                min_sim = max(sim_list)  # 最大類似度を取得

            # 判定
            if min_sim < sim_threshold:
                # 新規フレームとして保存
                out_filename = f"{video_basename}_{frame_idx}.jpg"
                out_path = os.path.join(out_dir, out_filename)
                cv2.imwrite(out_path, frame)

                # embed_listに追加
                embed_list.append(embed)

                added_count += 1

            pbar.update(1)

        pbar.close()
        cap.release()
        cv2.destroyAllWindows()

        print(f"[INFO] '{video_file}' -> New frames added: {added_count}\n")
        total_added_count += added_count

    print("="*50)
    print(f"処理が完了しました。")
    print(f"フォルダ '{video_dir}' 内の動画から新たに追加されたフレーム: {total_added_count} 枚")
    print(f"最終的な out_dir('{out_dir}') の合計画像数: {len(embed_list)} 枚")
    print("="*50)


if __name__ == "__main__":
    main()
