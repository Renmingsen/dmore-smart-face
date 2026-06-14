"""DMORE 智能脸谱 - FastAPI 后端（第一期核心模块）。
启动：./venv/bin/python -m uvicorn server.main:app --port 8800
"""
import os, io, json, shutil, csv, tempfile, time
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Body, Query
from fastapi.responses import Response, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from server import core

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(HERE, "web")
DEFAULT_DIR = "/path/to/your/photos"
OUTPUT_FOLDERS = {"工厂生产场景", "模特照", "模特图", "输出", "场景结果", "某人", "待删除"}

# 内容分布 / 标签的零样本类别
CONTENT_LABELS = {
    "产品图": "白底产品图 单个商品展示",
    "使用场景": "运动员比赛训练 使用产品的真实场景",
    "模特展示": "模特穿戴运动护具和球衣展示产品",
    "工厂生产": "工厂车间 生产线 工人作业 机器设备",
    "设施安装": "场地设施 拼装地板 球门 安装",
    "含文字海报": "带文字说明的海报 介绍图 排版",
}

app = FastAPI(title="DMORE 智能脸谱")
_label_vecs = None


def label_vecs():
    global _label_vecs
    if _label_vecs is None:
        _label_vecs = {k: core.text_vec(v) for k, v in CONTENT_LABELS.items()}
    return _label_vecs


def item(path, score=None, t=None):
    q = "?path=" + _q(path) + "&size=256" + (f"&t={t}" if t is not None else "")
    d = {"path": path, "name": os.path.basename(path), "thumb": "/thumb" + q}
    if score is not None:
        d["score"] = round(float(score), 3)
    if t is not None:
        d["t"] = round(float(t), 1)
    return d


def _q(s):
    import urllib.parse
    return urllib.parse.quote(s)


# ---------------- 文件/缩略图 ----------------
@app.get("/thumb")
def thumb(path: str, size: int = 256, t: float = None):
    if not core.safe(path):
        return Response(status_code=403)
    return Response(core.thumb_bytes(path, size, t), media_type="image/jpeg")


@app.get("/file")
def file(path: str):
    if not core.safe(path) or not os.path.isfile(path):
        return Response(status_code=404)
    return FileResponse(path)


# ---------------- 仪表盘统计 ----------------
@app.get("/api/stats")
def stats(dir: str = DEFAULT_DIR):
    paths, feats = core.image_index(dir, exclude=OUTPUT_FOLDERS)
    vids = core.list_videos(dir)
    dist = {}
    if len(feats):
        lv = label_vecs()
        M = np.stack([lv[k] for k in CONTENT_LABELS])  # [L,512]
        sims = feats @ M.T                              # [N,L]
        arg = sims.argmax(1)
        keys = list(CONTENT_LABELS)
        for i, k in enumerate(keys):
            dist[k] = int((arg == i).sum())
    # 人脸数（若已建索引）
    face_n = None
    fkey = os.path.join(core.CACHE, f"faceemb_{core.key(dir)}.npz")
    if os.path.exists(fkey):
        _, data = core.face_index(dir, exclude=OUTPUT_FOLDERS)
        face_n = sum(1 for i in range(len(paths)) if len(data[f"e{i}"]) > 0)
    return {"images": len(paths), "videos": len(vids), "faces": face_n,
            "distribution": dist}


# ---------------- 语义搜索 ----------------
@app.post("/api/search/scene")
def search_scene(dir: str = Body(DEFAULT_DIR), query: str = Body(...),
                 threshold: float = Body(0.36), limit: int = Body(150),
                 exclude_outputs: bool = Body(True)):
    ex = OUTPUT_FOLDERS if exclude_outputs else None
    paths, feats = core.image_index(dir, exclude=ex)
    if not len(feats):
        return {"total": 0, "items": []}
    tf = core.text_vec(query)
    sims = feats @ tf
    order = np.argsort(-sims)
    hit = [i for i in order if sims[i] >= threshold]
    items = [item(paths[i], sims[i]) for i in hit[:limit]]
    return {"total": len(hit), "shown": len(items), "items": items}


# ---------------- 以图搜图 ----------------
@app.post("/api/search/similar")
async def search_similar(dir: str = Form(DEFAULT_DIR), path: str = Form(None),
                         threshold: float = Form(0.6), limit: int = Form(60),
                         file: UploadFile = File(None)):
    paths, feats = core.image_index(dir, exclude=OUTPUT_FOLDERS)
    if not len(feats):
        return {"total": 0, "items": []}
    if file is not None:
        im = Image.open(io.BytesIO(await file.read())).convert("RGB")
        qv = core.image_vec(im)
    elif path and core.safe(path):
        im = core.load_rgb(path); qv = core.image_vec(im)
    else:
        return JSONResponse({"error": "需要 path 或上传图片"}, status_code=400)
    sims = feats @ qv
    order = np.argsort(-sims)
    hit = [i for i in order if sims[i] >= threshold and paths[i] != path]
    items = [item(paths[i], sims[i]) for i in hit[:limit]]
    return {"total": len(hit), "shown": len(items), "items": items}


# ---------------- 人物图谱（聚类） ----------------
@app.post("/api/people")
def people(dir: str = Body(DEFAULT_DIR), thr: float = Body(0.5),
           min_size: int = Body(2)):
    paths, data = core.face_index(dir, exclude=OUTPUT_FOLDERS)
    clusters = core.cluster_faces(paths, data, thr=thr)
    out = []
    for ci, c in enumerate(clusters):
        members = c["members"]
        if len(members) < min_size:
            continue
        cover = members[0][0]
        out.append({"id": ci, "size": len(members), "cover": item(cover)})
    return {"count": len(out), "people": out, "clusters_thr": thr}


@app.post("/api/people/photos")
def people_photos(dir: str = Body(DEFAULT_DIR), id: int = Body(...), thr: float = Body(0.5)):
    paths, data = core.face_index(dir, exclude=OUTPUT_FOLDERS)
    clusters = core.cluster_faces(paths, data, thr=thr)
    if id >= len(clusters):
        return {"items": []}
    seen, items = set(), []
    for p, _ in clusters[id]["members"]:
        if p not in seen:
            seen.add(p); items.append(item(p))
    return {"count": len(items), "items": items}


# ---------------- 找某个人 ----------------
@app.post("/api/find_person")
async def find_person(dir: str = Form(DEFAULT_DIR), threshold: float = Form(0.35),
                      files: list[UploadFile] = File(...)):
    appf = core.get_face()
    embs = []
    for f in files:
        im = Image.open(io.BytesIO(await f.read())).convert("RGB")
        faces = appf.get(core.to_bgr(im))
        if faces:
            big = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
            embs.append(big.normed_embedding)
    if not embs:
        return JSONResponse({"error": "参考图未检测到人脸"}, status_code=400)
    ref = np.mean(embs, 0); ref = ref / np.linalg.norm(ref)
    paths, data = core.face_index(dir, exclude=OUTPUT_FOLDERS)
    res = []
    for i, p in enumerate(paths):
        e = data[f"e{i}"]
        if len(e) == 0: continue
        best = float(np.max(e @ ref))
        if best >= threshold:
            res.append((best, p))
    res.sort(reverse=True)
    items = [item(p, s) for s, p in res]
    return {"total": len(items), "refs": len(embs), "items": items}


# ---------------- 智能标签（零样本分类） ----------------
@app.post("/api/tags")
def tags(dir: str = Body(DEFAULT_DIR), labels: dict = Body(None)):
    paths, feats = core.image_index(dir, exclude=OUTPUT_FOLDERS)
    L = labels or CONTENT_LABELS
    vecs = {k: core.text_vec(v) for k, v in L.items()}
    M = np.stack([vecs[k] for k in L]); keys = list(L)
    sims = feats @ M.T; arg = sims.argmax(1)
    groups = {k: [] for k in keys}
    for i in range(len(paths)):
        groups[keys[arg[i]]].append((float(sims[i][arg[i]]), paths[i]))
    out = {}
    for k in keys:
        g = sorted(groups[k], reverse=True)
        out[k] = {"count": len(g), "samples": [item(p, s) for s, p in g[:30]]}
    return {"labels": out}


# ---------------- 视频检索 ----------------
@app.post("/api/video/search")
def video_search(dir: str = Body(DEFAULT_DIR), query: str = Body(...),
                 threshold: float = Body(0.30), limit: int = Body(60),
                 every: float = Body(2.0), index_limit: int = Body(None)):
    vi = core.video_index(dir, every=every, limit=index_limit)
    if vi is None or not len(vi["feats"]):
        return {"total": 0, "items": [], "note": "该目录无视频或未建索引"}
    tf = core.text_vec(query)
    sims = vi["feats"] @ tf
    # 每个视频取最高分的时间点
    best = {}
    for k in range(len(sims)):
        v = int(vi["vidx"][k]); s = float(sims[k])
        if v not in best or s > best[v][0]:
            best[v] = (s, float(vi["ts"][k]))
    rows = [(s, t, vi["videos"][v]) for v, (s, t) in best.items() if s >= threshold]
    rows.sort(reverse=True)
    items = [item(vp, s, t) for s, t, vp in rows[:limit]]
    return {"total": len(rows), "items": items}


# ---------------- 导出（复制/移动+还原清单） ----------------
@app.post("/api/export")
def export(paths: list[str] = Body(...), dest: str = Body(...),
           mode: str = Body("copy")):
    if not core.safe(dest):
        return JSONResponse({"error": "目标路径不允许"}, status_code=400)
    os.makedirs(dest, exist_ok=True)
    rows, done = [], 0
    for rank, p in enumerate(paths, 1):
        if not os.path.isfile(p): continue
        base = os.path.basename(p)
        dst = os.path.join(dest, f"{rank:04d}_{base}"); n = 1
        while os.path.exists(dst):
            stem, ext = os.path.splitext(base); dst = os.path.join(dest, f"{rank:04d}_{stem}__{n}{ext}"); n += 1
        if mode == "move":
            shutil.move(p, dst); rows.append([dst, p])
        else:
            shutil.copy2(p, dst)
        done += 1
    msg = f"{'移动' if mode=='move' else '复制'} {done} 个到 {dest}"
    if rows:
        man = os.path.join(dest, "_还原清单.csv")
        with open(man, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f); w.writerow(["现位置", "原始位置"]); w.writerows(rows)
        msg += f"；还原清单 {man}"
    return {"ok": True, "done": done, "message": msg}


# ---------------- 去重清理 ----------------
@app.post("/api/dedup")
def dedup(dir: str = Body(DEFAULT_DIR), sim: float = Body(0.93), max_groups: int = Body(80)):
    paths, feats = core.image_index(dir, exclude=OUTPUT_FOLDERS)
    n = len(feats)
    if n == 0:
        return {"groups": [], "dup_extra": 0}
    S = feats @ feats.T
    np.fill_diagonal(S, 0)
    seen, groups = set(), []
    for i in range(n):
        if i in seen:
            continue
        js = [int(j) for j in np.where(S[i] >= sim)[0] if j not in seen and j != i]
        if js:
            grp = [i] + js
            for j in grp:
                seen.add(j)
            groups.append(grp)
            if len(groups) >= max_groups:
                break
    out = [{"size": len(g), "items": [item(paths[k]) for k in g]} for g in groups]
    return {"groups": out, "groups_n": len(out), "dup_extra": sum(len(g) - 1 for g in groups)}


# ---------------- 质量筛选 ----------------
@app.post("/api/quality")
def quality(dir: str = Body(DEFAULT_DIR), kind: str = Body("blur"), limit: int = Body(150)):
    paths, q = core.quality_index(dir, exclude=OUTPUT_FOLDERS)
    if not paths:
        return {"items": [], "note": "无图片"}
    arr = np.array(q, np.float32)
    sharp, bright = arr[:, 0], arr[:, 1]
    if kind == "blur":
        idx = np.argsort(sharp)[:limit]
    elif kind == "sharp":
        idx = np.argsort(-sharp)[:limit]
    elif kind == "dark":
        idx = np.where(bright < 60)[0]
    elif kind == "bright":
        idx = np.where(bright > 205)[0]
    else:
        idx = np.argsort(sharp)[:limit]
    items = [item(paths[i], float(sharp[i])) for i in idx[:limit]]
    return {"total": len(items), "items": items}


# ---------------- 自动相册（KMeans 聚类） ----------------
_album_cache = {}
@app.post("/api/albums")
def albums(dir: str = Body(DEFAULT_DIR), k: int = Body(12)):
    paths, feats = core.image_index(dir, exclude=OUTPUT_FOLDERS)
    if not len(feats):
        return {"albums": []}
    assign, cen = core.kmeans(feats, k)
    _album_cache[dir] = (paths, assign)
    lv = label_vecs(); keys = list(CONTENT_LABELS); M = np.stack([lv[x] for x in keys])
    out = []
    for j in range(len(cen)):
        mem = np.where(assign == j)[0]
        if not len(mem):
            continue
        lab = keys[int((cen[j] @ M.T).argmax())]
        sims = feats[mem] @ cen[j]
        cover = paths[int(mem[sims.argmax()])]
        out.append({"id": int(j), "size": int(len(mem)), "label": lab, "cover": item(cover)})
    out.sort(key=lambda x: -x["size"])
    return {"albums": out}


@app.post("/api/albums/photos")
def albums_photos(dir: str = Body(DEFAULT_DIR), id: int = Body(...)):
    if dir not in _album_cache:
        return {"items": []}
    paths, assign = _album_cache[dir]
    items = [item(paths[i]) for i in np.where(assign == id)[0]]
    return {"count": len(items), "items": items}


# ---------------- 同框关系 ----------------
@app.post("/api/cooccur")
def cooccur(dir: str = Body(DEFAULT_DIR), thr: float = Body(0.5)):
    from collections import Counter
    paths, data = core.face_index(dir, exclude=OUTPUT_FOLDERS)
    clusters = core.cluster_faces(paths, data, thr=thr)
    img_cl = {}
    for cid, c in enumerate(clusters):
        for p, _ in c["members"]:
            img_cl.setdefault(p, set()).add(cid)
    pair = Counter()
    for cs in img_cl.values():
        cs = sorted(cs)
        for a in range(len(cs)):
            for b in range(a + 1, len(cs)):
                pair[(cs[a], cs[b])] += 1
    res = []
    for (a, b), nn in pair.most_common(30):
        res.append({"a": a, "b": b, "count": nn,
                    "acover": item(clusters[a]["members"][0][0]),
                    "bcover": item(clusters[b]["members"][0][0])})
    return {"pairs": res, "people": len(clusters)}


# ---------------- 视频分镜 ----------------
@app.post("/api/video/shots")
def video_shots(path: str = Body(...)):
    import cv2
    if not core.safe(path) or not os.path.isfile(path):
        return JSONResponse({"error": "视频不存在"}, status_code=404)
    shots = core.detect_shots(path)
    for s in shots:
        s["thumb"] = "/thumb?path=" + _q(path) + "&size=240&t=" + str(s["mid"])
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    info = {"name": os.path.basename(path),
            "dur": round((cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps, 1),
            "w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "shots": len(shots)}
    cap.release()
    return {"shots": shots[:80], "info": info}


# ---------------- 视频片段导出 ----------------
@app.post("/api/video/clip")
def video_clip(path: str = Body(...), start: float = Body(...), end: float = Body(...),
               dest: str = Body(None)):
    if not core.safe(path) or not os.path.isfile(path):
        return JSONResponse({"error": "视频不存在"}, status_code=404)
    base = os.path.splitext(os.path.basename(path))[0]
    dest = dest or os.path.join(os.path.dirname(path), "_片段", f"{base}_{start:.0f}-{end:.0f}.mp4")
    try:
        out = core.cut_clip(path, start, end, dest)
    except Exception as e:
        return JSONResponse({"error": str(e)[:200]}, status_code=500)
    return {"ok": True, "out": out, "message": f"已剪出片段：{out}"}


# ---------------- 省电模式 / 缓存 ----------------
@app.get("/api/power")
def power_get():
    return {"power_save": core._config.get("power_save", False), "ncpu": core.NCPU}


@app.post("/api/power")
def power_set(save: bool = Body(..., embed=True)):
    core.set_power(save)
    n = max(1, core.NCPU // 2) if save else core.NCPU
    return {"power_save": save, "threads": n, "ncpu": core.NCPU}


@app.get("/api/cache/info")
def cache_info():
    return core.cache_info()


@app.post("/api/cache/clear")
def cache_clear(thumbs_only: bool = Body(False, embed=True)):
    n = core.cache_clear(thumbs_only)
    return {"ok": True, "removed": n, "message": f"已清理缓存（{n} 项），索引会在下次使用时自动重建。"}


# 前端静态资源（放最后，避免覆盖 /api）
app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
