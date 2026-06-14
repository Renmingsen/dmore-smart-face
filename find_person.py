"""
在照片堆里找某个特定人物（人脸识别，InsightFace / ArcFace）。
跨时间、换衣服、不同光线都能认，因为只比对『脸』。

准备：在 references/ 文件夹里放 1~3 张这个人的清晰正脸照片（jpg/png 都行）。

用法：
  # 第一次会建人脸索引并缓存，之后秒搜
  python find_person.py --photos /path/to/photos --refs references --threshold 0.35

结果复制到 results/person/ ，文件名前缀是相似度（越高越像）。
阈值建议：0.3 偏宽松（多召回，可能有误判），0.4 偏严格（更准，可能漏）。先用 0.35 看效果再调。
"""
import argparse
import hashlib
import json
import os
import shutil

import numpy as np
from tqdm import tqdm

from insightface.app import FaceAnalysis

from pf_common import list_images, load_rgb

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "cache")


def pil_to_bgr(img):
    """PIL RGB -> numpy BGR（InsightFace 要 BGR）。"""
    arr = np.asarray(img)[:, :, ::-1].copy()
    return arr


def get_app():
    # CPU 跑，稳定；buffalo_l 首次会下载 ~300MB
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def build_ref_embedding(app, refs_dir):
    refs = list_images(refs_dir)
    if not refs:
        raise SystemExit(f"请先在 {refs_dir} 放 1~3 张目标人物的清晰正脸照片")
    embs = []
    for p in refs:
        img = load_rgb(p)
        if img is None:
            continue
        faces = app.get(pil_to_bgr(img))
        if not faces:
            print(f"  [警告] 参考图没检测到人脸：{p}")
            continue
        # 取最大的脸
        f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        embs.append(f.normed_embedding)
    if not embs:
        raise SystemExit("参考图里一张脸都没检测到，请换更清晰的正脸照片")
    ref = np.mean(embs, axis=0)
    ref = ref / np.linalg.norm(ref)
    print(f"参考脸：用了 {len(embs)} 张照片合成特征")
    return ref


def build_or_load_face_index(app, photos_dir):
    """每张照片提取所有人脸的特征；缓存到磁盘。
    返回 dict: path -> list[embedding(512,)]"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = hashlib.md5(os.path.abspath(photos_dir).encode()).hexdigest()[:12]
    npz_file = os.path.join(CACHE_DIR, f"faces_{key}.npz")
    path_file = os.path.join(CACHE_DIR, f"faces_{key}.json")

    all_paths = list_images(photos_dir)
    if not all_paths:
        raise SystemExit(f"在 {photos_dir} 下没找到任何图片")

    if os.path.exists(npz_file) and os.path.exists(path_file):
        cached = json.load(open(path_file, encoding="utf-8"))
        if cached == all_paths:
            print(f"复用人脸缓存：{len(all_paths)} 张")
            data = np.load(npz_file, allow_pickle=True)
            return {p: list(data[str(i)]) for i, p in enumerate(all_paths)}
        print("照片有变动，重建人脸索引…")

    print(f"开始检测 {len(all_paths)} 张图的人脸（首次较慢，之后走缓存）")
    result = {}
    save = {}
    for i, p in enumerate(tqdm(all_paths, desc="检测人脸")):
        img = load_rgb(p)
        if img is None:
            result[p] = []
            save[str(i)] = np.zeros((0, 512), dtype=np.float32)
            continue
        faces = app.get(pil_to_bgr(img))
        embs = [f.normed_embedding.astype(np.float32) for f in faces]
        result[p] = embs
        save[str(i)] = np.array(embs, dtype=np.float32) if embs else np.zeros((0, 512), dtype=np.float32)

    np.savez_compressed(npz_file, **save)
    json.dump(all_paths, open(path_file, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"人脸索引已缓存 -> {npz_file}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", required=True, help="照片根目录")
    ap.add_argument("--refs", default=os.path.join(HERE, "references"), help="参考正脸目录")
    ap.add_argument("--threshold", type=float, default=0.35, help="相似度阈值，0.3宽松~0.4严格")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "person"))
    args = ap.parse_args()

    print("加载人脸模型 buffalo_l …")
    app = get_app()

    ref = build_ref_embedding(app, args.refs)
    face_index = build_or_load_face_index(app, args.photos)

    # 每张图：取所有脸里和参考脸最高的相似度
    hits = []
    for path, embs in face_index.items():
        if not embs:
            continue
        sims = [float(np.dot(ref, e)) for e in embs]
        best = max(sims)
        if best >= args.threshold:
            hits.append((best, path))
    hits.sort(reverse=True)

    os.makedirs(args.out, exist_ok=True)
    print(f"\n命中 {len(hits)} 张（阈值 {args.threshold}），复制到：{args.out}")
    for rank, (score, src) in enumerate(hits, 1):
        name = f"{score:.3f}_{rank:04d}_{os.path.basename(src)}"
        shutil.copy2(src, os.path.join(args.out, name))
    if hits:
        print(f"最像 {hits[0][0]:.3f}，最低 {hits[-1][0]:.3f}。文件名前缀=相似度，越高越像。")
        print("若误判多 -> 调高 --threshold（如 0.4）；若漏太多 -> 调低（如 0.3）。")
    else:
        print("一张都没命中：可能阈值太高，或参考脸不够清晰。试试 --threshold 0.28")


if __name__ == "__main__":
    main()
