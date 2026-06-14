"""
按场景/语义搜索照片（中文 CLIP）。

用法：
  # 第一次：建索引（把一万张图编码成向量，会缓存，下次秒开）
  python scene_search.py --photos /path/to/photos --query "工厂生产场景 车间 流水线" --topk 200

  # 之后换关键词，直接复用缓存：
  python scene_search.py --photos /path/to/photos --query "户外 阳光" --topk 100

结果会把最匹配的若干张复制到 results/scene_xxx/ ，文件名带相似度排名，方便直接看。
"""
import argparse
import hashlib
import json
import os
import shutil

# 国内用 HuggingFace 镜像下载模型，否则极慢/卡住（必须在导入 cn_clip 前设置）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
from tqdm import tqdm

import cn_clip.clip as clip
from cn_clip.clip import load_from_name

from pf_common import list_images, load_rgb

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "cache")
MODEL_DIR = os.path.join(HERE, "models")


def pick_device(name):
    if name == "cpu":
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_or_load_index(photos_dir, model, preprocess, device, batch=16):
    """返回 (paths, feats[N,512])，对图片向量做磁盘缓存。"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = hashlib.md5(os.path.abspath(photos_dir).encode()).hexdigest()[:12]
    feat_file = os.path.join(CACHE_DIR, f"scene_{key}.npy")
    path_file = os.path.join(CACHE_DIR, f"scene_{key}.json")

    all_paths = list_images(photos_dir)
    if not all_paths:
        raise SystemExit(f"在 {photos_dir} 下没找到任何图片")

    # 已有缓存且照片集合没变 -> 直接用
    if os.path.exists(feat_file) and os.path.exists(path_file):
        cached = json.load(open(path_file, encoding="utf-8"))
        if cached == all_paths:
            print(f"复用缓存索引：{len(all_paths)} 张")
            return all_paths, np.load(feat_file)
        print("照片有变动，重建索引…")

    print(f"开始编码 {len(all_paths)} 张图片（首次较慢，之后走缓存）")
    feats = np.zeros((len(all_paths), 512), dtype=np.float32)
    buf_imgs, buf_idx = [], []

    def flush():
        if not buf_imgs:
            return
        x = torch.stack(buf_imgs).to(device)
        with torch.no_grad():
            f = model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
        feats[buf_idx] = f.cpu().numpy()
        buf_imgs.clear()
        buf_idx.clear()

    for i, p in enumerate(tqdm(all_paths, desc="编码")):
        img = load_rgb(p)
        if img is None:
            continue
        buf_imgs.append(preprocess(img))
        buf_idx.append(i)
        if len(buf_imgs) >= batch:
            flush()
    flush()

    np.save(feat_file, feats)
    json.dump(all_paths, open(path_file, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"索引已缓存 -> {feat_file}")
    return all_paths, feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", required=True, help="照片根目录")
    ap.add_argument("--query", required=True, help="中文描述，如『工厂生产场景 车间』")
    ap.add_argument("--topk", type=int, default=200, help="挑出前多少张")
    ap.add_argument("--threshold", type=float, default=None,
                    help="按相似度阈值筛选（给了就用阈值，忽略 topk）；越高越严")
    ap.add_argument("--stats", action="store_true", help="只看分数分布，不复制")
    ap.add_argument("--out", default=None, help="结果输出目录")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    ap.add_argument("--copy", action="store_true", default=True, help="复制结果（默认开）")
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"设备：{device}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("加载中文 CLIP 模型（首次会下载 ~400MB）…")
    # use_modelscope=True 从魔搭(国内)下载，避免 HuggingFace 卡住
    model, preprocess = load_from_name("ViT-B-16", device=device, download_root=MODEL_DIR, use_modelscope=True)
    model.eval()

    paths, feats = build_or_load_index(args.photos, model, preprocess, device)

    # 文本向量
    text = clip.tokenize([args.query]).to(device)
    with torch.no_grad():
        tf = model.encode_text(text)
        tf = tf / tf.norm(dim=-1, keepdim=True)
    tf = tf.cpu().numpy()[0]

    sims = feats @ tf  # 余弦相似度
    full_order = np.argsort(-sims)

    # 分数分布（帮你定阈值）
    pcts = [99, 98, 95, 90, 80, 50]
    print("\n相似度分布（百分位）：")
    for p in pcts:
        print(f"  前{100 - p:>2}% 的分数线 ≈ {np.percentile(sims, p):.3f}")
    print(f"  最高分 {sims.max():.3f}")

    if args.stats:
        return

    if args.threshold is not None:
        order = [i for i in full_order if sims[i] >= args.threshold]
        print(f"\n相似度 ≥ {args.threshold} 的有 {len(order)} 张")
    else:
        order = list(full_order[: args.topk])
        print(f"\n取前 {args.topk} 张")

    out = args.out or os.path.join(HERE, "results", "scene_" + args.query.replace(" ", "_")[:20])
    os.makedirs(out, exist_ok=True)
    for rank, idx in enumerate(order, 1):
        src = paths[idx]
        score = sims[idx]
        name = f"{rank:04d}_{score:.3f}_{os.path.basename(src)}"
        shutil.copy2(src, os.path.join(out, name))
    if order:
        print(f"已复制到：{out}")
        print(f"最高分 {sims[order[0]]:.3f}，最低分 {sims[order[-1]]:.3f}。文件名前缀=排名_相似度。")


if __name__ == "__main__":
    main()
