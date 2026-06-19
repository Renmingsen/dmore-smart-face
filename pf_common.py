# DMORE 智能脸谱 (DMORE Vision) — 本地识图工作站
# Copyright (C) 2026 DMORE / Renmingsen
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See <https://www.gnu.org/licenses/>.

"""照片筛选 - 公共工具：图片遍历、HEIC 支持、读图。"""
import os
from PIL import Image

# 让 PIL 能读 iPhone 的 HEIC/HEIF（若已安装 pillow-heif）
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_OK = True
except Exception:
    HEIC_OK = False

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".heic", ".heif"}


def list_images(root):
    """递归列出 root 下所有图片的绝对路径（排序，稳定）。"""
    paths = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in IMG_EXTS:
                paths.append(os.path.join(dirpath, f))
    paths.sort()
    return paths


def load_rgb(path):
    """读成 RGB 的 PIL.Image，失败返回 None。"""
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"  [跳过] 无法读取 {path}: {e}")
        return None
