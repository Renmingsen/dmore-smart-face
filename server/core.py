# DMORE 智能脸谱 (DMORE Vision) — 本地识图工作站
# Copyright (C) 2026 DMORE / Renmingsen
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See <https://www.gnu.org/licenses/>.

"""DMORE 智能脸谱 - 核心引擎与索引层。
复用本地 CLIP(cn_clip) + InsightFace，新增视频抽帧索引、人脸聚类、缩略图缓存。
"""
import os, io, json, hashlib, time, glob
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import numpy as np
import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
try:
    import pillow_heif; pillow_heif.register_heif_opener()
except Exception:
    pass
import cv2
import cn_clip.clip as clip
from cn_clip.clip import load_from_name
from insightface.app import FaceAnalysis

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(HERE, "cache")
MODELS = os.path.join(HERE, "models")
THUMBS = os.path.join(CACHE, "thumbs")
os.makedirs(THUMBS, exist_ok=True)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".heic", ".heif"}
VID_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".flv", ".wmv"}
ALLOWED_ROOTS = ["/Volumes", "/Users", "/tmp", "/private"]

_clip = None
_face = None


def get_clip():
    global _clip
    if _clip is None:
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        m, prep = load_from_name("ViT-B-16", device=dev, download_root=MODELS, use_modelscope=True)
        m.eval()
        _clip = (m, prep, dev)
    return _clip


def get_face():
    global _face
    if _face is None:
        a = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        a.prepare(ctx_id=-1, det_size=(640, 640))
        _face = a
    return _face


def safe(path):
    p = os.path.abspath(path)
    return any(p.startswith(r) for r in ALLOWED_ROOTS)


def key(d):
    return hashlib.md5(os.path.abspath(d).encode()).hexdigest()[:12]


def list_images(root, exclude=None):
    exclude = set(exclude or [])
    out = []
    for dp, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in exclude]
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                out.append(os.path.join(dp, f))
    out.sort()
    return out


def list_videos(root):
    out = []
    for dp, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() in VID_EXTS:
                out.append(os.path.join(dp, f))
    out.sort()
    return out


def load_rgb(p):
    try:
        return Image.open(p).convert("RGB")
    except Exception:
        return None


def to_bgr(im, maxside=1600):
    im = im.convert("RGB"); w, h = im.size; mx = max(w, h)
    if mx > maxside:
        s = maxside / mx; im = im.resize((int(w*s), int(h*s)))
    return np.asarray(im)[:, :, ::-1].copy()


# ---------- 缩略图缓存 ----------
def thumb_bytes(path, size=256, t=None):
    """返回 jpeg 字节。t 给定则取视频该秒的帧。磁盘缓存。"""
    tag = hashlib.md5(f"{path}|{size}|{t}".encode()).hexdigest()[:16]
    cf = os.path.join(THUMBS, tag + ".jpg")
    if os.path.exists(cf):
        return open(cf, "rb").read()
    im = None
    if t is not None or os.path.splitext(path)[1].lower() in VID_EXTS:
        cap = cv2.VideoCapture(path)
        if t:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000)
        ok, fr = cap.read(); cap.release()
        if ok:
            im = Image.fromarray(fr[:, :, ::-1])
    else:
        im = load_rgb(path)
    if im is None:
        im = Image.new("RGB", (size, size), (60, 60, 66))
    im.thumbnail((size, size))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=82)
    data = buf.getvalue()
    open(cf, "wb").write(data)
    return data


# ---------- 图像 CLIP 索引 ----------
def image_index(d, exclude=None, progress=None):
    ff = os.path.join(CACHE, f"scene_{key(d)}.npy")
    pf = os.path.join(CACHE, f"scene_{key(d)}.json")
    paths = list_images(d, exclude)
    if not paths:
        return [], np.zeros((0, 512), np.float32)
    if os.path.exists(ff) and os.path.exists(pf) and json.load(open(pf, encoding="utf-8")) == paths:
        return paths, np.load(ff)
    m, prep, dev = get_clip()
    feats = np.zeros((len(paths), 512), np.float32)
    buf, idx = [], []
    def flush():
        if not buf: return
        x = torch.stack(buf).to(dev)
        with torch.no_grad():
            f = m.encode_image(x); f = f / f.norm(dim=-1, keepdim=True)
        feats[idx] = f.cpu().numpy(); buf.clear(); idx.clear()
    for i, p in enumerate(paths):
        im = load_rgb(p)
        if im is None: continue
        buf.append(prep(im)); idx.append(i)
        if len(buf) >= 16: flush()
        if progress and i % 50 == 0: progress(i, len(paths))
    flush()
    np.save(ff, feats); json.dump(paths, open(pf, "w", encoding="utf-8"), ensure_ascii=False)
    return paths, feats


def text_vec(q):
    m, prep, dev = get_clip()
    t = clip.tokenize([q]).to(dev)
    with torch.no_grad():
        tf = m.encode_text(t); tf = tf / tf.norm(dim=-1, keepdim=True)
    return tf.cpu().numpy()[0]


def image_vec(im):
    m, prep, dev = get_clip()
    x = prep(im).unsqueeze(0).to(dev)
    with torch.no_grad():
        f = m.encode_image(x); f = f / f.norm(dim=-1, keepdim=True)
    return f.cpu().numpy()[0]


# ---------- 人脸索引(embedding + 占比/置信度) ----------
def face_index(d, exclude=None, progress=None):
    npz = os.path.join(CACHE, f"faceemb_{key(d)}.npz")
    pf = os.path.join(CACHE, f"faceemb_{key(d)}.json")
    paths = list_images(d, exclude)
    if not paths:
        return [], {}
    if os.path.exists(npz) and os.path.exists(pf) and json.load(open(pf, encoding="utf-8")) == paths:
        return paths, np.load(npz, allow_pickle=True)
    app = get_face()
    save = {}
    for i, p in enumerate(paths):
        im = load_rgb(p)
        if im is None:
            save[f"e{i}"] = np.zeros((0, 512), np.float32); save[f"m{i}"] = np.array([0, 0], np.float32)
        else:
            arr = to_bgr(im); faces = app.get(arr)
            if faces:
                embs = np.array([f.normed_embedding for f in faces], np.float32)
                ah, aw = arr.shape[:2]
                def ar(fc):
                    x1, y1, x2, y2 = fc.bbox; return float(max(0, x2-x1)*max(0, y2-y1))
                prom = max(ar(f) for f in faces)/float(aw*ah)
                conf = float(max(f.det_score for f in faces))
                save[f"e{i}"] = embs; save[f"m{i}"] = np.array([prom, conf], np.float32)
            else:
                save[f"e{i}"] = np.zeros((0, 512), np.float32); save[f"m{i}"] = np.array([0, 0], np.float32)
        if progress and i % 25 == 0: progress(i, len(paths))
    np.savez_compressed(npz, **save)
    json.dump(paths, open(pf, "w", encoding="utf-8"), ensure_ascii=False)
    return paths, np.load(npz, allow_pickle=True)


def cluster_faces(paths, data, thr=0.5, min_conf=0.6):
    """贪心聚类：把所有人脸 embedding 按相似度并入已有簇。返回簇列表。"""
    clusters = []  # {centroid, members:[(path, emb)]}
    for i, p in enumerate(paths):
        embs = data[f"e{i}"]; m = data[f"m{i}"]
        if len(embs) == 0: continue
        conf = float(m[1]) if len(m) > 1 else 1.0
        if conf < min_conf: continue
        for e in embs:
            best, bj = thr, -1
            for j, c in enumerate(clusters):
                s = float(np.dot(c["centroid"], e))
                if s > best: best, bj = s, j
            if bj >= 0:
                c = clusters[bj]; c["members"].append((p, e))
                cen = c["centroid"] * c["n"] + e; c["n"] += 1
                c["centroid"] = cen / np.linalg.norm(cen)
            else:
                clusters.append({"centroid": e.copy(), "n": 1, "members": [(p, e)]})
    clusters.sort(key=lambda c: -len(c["members"]))
    return clusters


# ---------- 视频抽帧索引 ----------
def video_index(d, every=2.0, progress=None, limit=None):
    """每 every 秒抽一帧，存 CLIP 向量 + (video_path, t)。缓存。"""
    npz = os.path.join(CACHE, f"video_{key(d)}.npz")
    jf = os.path.join(CACHE, f"video_{key(d)}.json")
    vids = list_videos(d)
    if limit: vids = vids[:limit]
    if not vids:
        return None
    if os.path.exists(npz) and os.path.exists(jf):
        meta = json.load(open(jf, encoding="utf-8"))
        if meta.get("videos") == vids:
            z = np.load(npz)
            return {"feats": z["feats"], "vidx": z["vidx"], "ts": z["ts"], "videos": vids}
    m, prep, dev = get_clip()
    feats, vidx, ts = [], [], []
    for vi, vp in enumerate(vids):
        cap = cv2.VideoCapture(vp)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        dur = n / fps if fps else 0
        t = 0.0
        batch, bts = [], []
        while t < dur:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, fr = cap.read()
            if not ok: break
            batch.append(prep(Image.fromarray(fr[:, :, ::-1]))); bts.append(t)
            if len(batch) >= 16:
                x = torch.stack(batch).to(dev)
                with torch.no_grad():
                    f = m.encode_image(x); f = f / f.norm(dim=-1, keepdim=True)
                for k in range(len(batch)):
                    feats.append(f[k].cpu().numpy()); vidx.append(vi); ts.append(bts[k])
                batch, bts = [], []
            t += every
        if batch:
            x = torch.stack(batch).to(dev)
            with torch.no_grad():
                f = m.encode_image(x); f = f / f.norm(dim=-1, keepdim=True)
            for k in range(len(batch)):
                feats.append(f[k].cpu().numpy()); vidx.append(vi); ts.append(bts[k])
        cap.release()
        if progress: progress(vi + 1, len(vids))
    feats = np.array(feats, np.float32) if feats else np.zeros((0, 512), np.float32)
    np.savez_compressed(npz, feats=feats, vidx=np.array(vidx, np.int32), ts=np.array(ts, np.float32))
    json.dump({"videos": vids}, open(jf, "w", encoding="utf-8"), ensure_ascii=False)
    return {"feats": feats, "vidx": np.array(vidx), "ts": np.array(ts), "videos": vids}


# ---------- 质量(清晰度)索引 ----------
def quality_index(d, exclude=None, progress=None):
    jf = os.path.join(CACHE, f"quality_{key(d)}.json")
    paths = list_images(d, exclude)
    if not paths:
        return [], []
    if os.path.exists(jf):
        data = json.load(open(jf, encoding="utf-8"))
        if data.get("paths") == paths:
            return paths, data["q"]
    q = []
    for i, p in enumerate(paths):
        im = load_rgb(p)
        if im is None:
            q.append([0.0, 0.0]); continue
        im2 = im.copy(); im2.thumbnail((900, 900))
        g = cv2.cvtColor(np.asarray(im2), cv2.COLOR_RGB2GRAY)
        sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())   # 清晰度
        bright = float(g.mean())                            # 亮度
        q.append([round(sharp, 1), round(bright, 1)])
        if progress and i % 200 == 0:
            progress(i, len(paths))
    json.dump({"paths": paths, "q": q}, open(jf, "w", encoding="utf-8"), ensure_ascii=False)
    return paths, q


# ---------- 轻量 KMeans(余弦) ----------
def kmeans(feats, k=12, iters=15, seed=0):
    rng = np.random.default_rng(seed)
    n = len(feats)
    k = min(k, n)
    cen = feats[rng.choice(n, k, replace=False)].copy()
    assign = np.zeros(n, np.int32)
    for _ in range(iters):
        sim = feats @ cen.T
        assign = sim.argmax(1)
        for j in range(k):
            m = feats[assign == j]
            if len(m):
                v = m.mean(0); cen[j] = v / (np.linalg.norm(v) + 1e-9)
    return assign, cen


# ---------- 视频镜头检测 ----------
def detect_shots(video, sample=0.5, thr=0.45, min_len=1.0):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps if fps else 0
    if dur <= 0:
        cap.release(); return []
    prev = None; cuts = [0.0]; t = 0.0
    while t < dur:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok: break
        h = cv2.calcHist([cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(h, h)
        if prev is not None:
            corr = cv2.compareHist(prev, h, cv2.HISTCMP_CORREL)
            if corr < (1 - thr) and (t - cuts[-1]) >= min_len:
                cuts.append(t)
        prev = h; t += sample
    cap.release()
    cuts.append(dur)
    shots = []
    for i in range(len(cuts) - 1):
        s, e = cuts[i], cuts[i + 1]
        if e - s >= min_len:
            shots.append({"start": round(s, 2), "end": round(e, 2), "mid": round((s + e) / 2, 2)})
    return shots


def cut_clip(video, start, end, dest):
    import subprocess
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-ss", str(start), "-to", str(end), "-i", video,
           "-c:v", "libx264", "-c:a", "aac", "-loglevel", "error", dest]
    subprocess.run(cmd, check=True)
    return dest


# ---------- 配置 / 省电模式 ----------
import multiprocessing
CONFIG_FILE = os.path.join(HERE, "config.json")
NCPU = os.cpu_count() or 8
_config = {"power_save": False}


def load_config():
    global _config
    try:
        _config.update(json.load(open(CONFIG_FILE, encoding="utf-8")))
    except Exception:
        pass
    apply_power(_config.get("power_save", False))
    return _config


def save_config():
    json.dump(_config, open(CONFIG_FILE, "w", encoding="utf-8"), ensure_ascii=False)


def apply_power(save):
    """省电模式：把计算线程数限制到约一半核心，CPU 占用减半（速度变慢）。"""
    global _face
    n = max(1, NCPU // 2) if save else NCPU
    try:
        torch.set_num_threads(n)
    except Exception:
        pass
    try:
        cv2.setNumThreads(n)
    except Exception:
        pass
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    _config["power_save"] = bool(save)
    _face = None   # 让人脸引擎下次以新线程数重建
    return n


def set_power(save):
    apply_power(save); save_config()
    return _config


def cache_info():
    total = 0
    for dp, _, files in os.walk(CACHE):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dp, f))
            except Exception:
                pass
    return {"bytes": total, "mb": round(total / 1024 / 1024, 1)}


def cache_clear(thumbs_only=False):
    import shutil
    removed = 0
    if thumbs_only:
        if os.path.isdir(THUMBS):
            shutil.rmtree(THUMBS); os.makedirs(THUMBS, exist_ok=True); removed = 1
    else:
        for f in os.listdir(CACHE):
            p = os.path.join(CACHE, f)
            try:
                if os.path.isdir(p):
                    shutil.rmtree(p)
                else:
                    os.remove(p)
                removed += 1
            except Exception:
                pass
        os.makedirs(THUMBS, exist_ok=True)
    return removed


load_config()
