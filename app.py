# DMORE 智能脸谱 (DMORE Vision) — 本地识图工作站
# Copyright (C) 2026 DMORE / Renmingsen
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See <https://www.gnu.org/licenses/>.

"""
照片识别系统 - 本地网页界面（Gradio）
启动： ./venv/bin/python app.py   然后浏览器打开 http://127.0.0.1:7860

三个功能：
  1) 场景搜索   —— 中文 CLIP，按内容找图（如「工厂生产场景」）
  2) 找特定人物 —— InsightFace 人脸识别，给参考脸找同一个人
  3) 人物照筛选 —— 人脸检测 + CLIP，挑出「人物+产品」的模特图
全部本地运行、免费。结果可一键「复制」或「移动（带还原清单）」到指定文件夹。
"""
import os, json, hashlib, shutil, csv, time
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import numpy as np
import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import gradio as gr
import cn_clip.clip as clip
from cn_clip.clip import load_from_name
from insightface.app import FaceAnalysis

from pf_common import list_images, load_rgb

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "cache")
MODEL_DIR = os.path.join(HERE, "models")
os.makedirs(CACHE_DIR, exist_ok=True)

# ---------- 模型懒加载 ----------
_clip = None
_face = None

def get_clip():
    global _clip
    if _clip is None:
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        m, prep = load_from_name("ViT-B-16", device=dev, download_root=MODEL_DIR, use_modelscope=True)
        m.eval()
        _clip = (m, prep, dev)
    return _clip

def get_face():
    global _face
    if _face is None:
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        _face = app
    return _face

def _key(photos_dir):
    return hashlib.md5(os.path.abspath(photos_dir).encode()).hexdigest()[:12]

def _to_bgr(im, maxside=1600):
    im = im.convert("RGB"); w, h = im.size; mx = max(w, h)
    if mx > maxside:
        s = maxside / mx; im = im.resize((int(w*s), int(h*s)))
    return np.asarray(im)[:, :, ::-1].copy()

# ---------- CLIP 图像索引（缓存） ----------
def build_clip_index(photos_dir, progress=None):
    m, prep, dev = get_clip()
    key = _key(photos_dir)
    ff = os.path.join(CACHE_DIR, f"scene_{key}.npy")
    pf = os.path.join(CACHE_DIR, f"scene_{key}.json")
    paths = list_images(photos_dir)
    if not paths:
        raise gr.Error(f"目录里没有图片：{photos_dir}")
    if os.path.exists(ff) and os.path.exists(pf) and json.load(open(pf, encoding="utf-8")) == paths:
        return paths, np.load(ff)
    feats = np.zeros((len(paths), 512), dtype=np.float32)
    buf, bidx = [], []
    def flush():
        if not buf: return
        x = torch.stack(buf).to(dev)
        with torch.no_grad():
            f = m.encode_image(x); f = f / f.norm(dim=-1, keepdim=True)
        feats[bidx] = f.cpu().numpy(); buf.clear(); bidx.clear()
    it = enumerate(paths)
    if progress is not None:
        it = progress.tqdm(list(it), desc="建立图像索引")
    for i, p in it:
        img = load_rgb(p)
        if img is None: continue
        buf.append(prep(img)); bidx.append(i)
        if len(buf) >= 16: flush()
    flush()
    np.save(ff, feats); json.dump(paths, open(pf, "w", encoding="utf-8"), ensure_ascii=False)
    return paths, feats

def clip_text_vec(query):
    m, prep, dev = get_clip()
    t = clip.tokenize([query]).to(dev)
    with torch.no_grad():
        tf = m.encode_text(t); tf = tf / tf.norm(dim=-1, keepdim=True)
    return tf.cpu().numpy()[0]

# ---------- 人脸索引（embedding + 占比/置信度，缓存） ----------
def build_face_index(photos_dir, progress=None):
    app = get_face()
    key = _key(photos_dir)
    npz = os.path.join(CACHE_DIR, f"faceemb_{key}.npz")
    pf = os.path.join(CACHE_DIR, f"faceemb_{key}.json")
    paths = list_images(photos_dir)
    if not paths:
        raise gr.Error(f"目录里没有图片：{photos_dir}")
    if os.path.exists(npz) and os.path.exists(pf) and json.load(open(pf, encoding="utf-8")) == paths:
        data = np.load(npz, allow_pickle=True)
        return paths, data
    save = {}
    it = enumerate(paths)
    if progress is not None:
        it = progress.tqdm(list(it), desc="检测人脸")
    for i, p in it:
        img = load_rgb(p)
        if img is None:
            save[f"e{i}"] = np.zeros((0, 512), np.float32); save[f"m{i}"] = np.array([0, 0], np.float32); continue
        arr = _to_bgr(img)
        faces = app.get(arr)
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
    np.savez_compressed(npz, **save)
    json.dump(paths, open(pf, "w", encoding="utf-8"), ensure_ascii=False)
    return paths, np.load(npz, allow_pickle=True)

# ---------- 导出（复制 / 移动+还原清单） ----------
def export(paths_scores, dest, mode):
    if not paths_scores:
        return "没有可导出的结果，请先搜索。"
    if not dest.strip():
        return "请填写目标文件夹路径。"
    os.makedirs(dest, exist_ok=True)
    rows = []
    done = 0
    for rank, (p, s) in enumerate(paths_scores, 1):
        if not os.path.isfile(p):
            continue
        base = os.path.basename(p)
        dst = os.path.join(dest, f"{rank:04d}_{s:.3f}_{base}")
        n = 1
        while os.path.exists(dst):
            stem, ext = os.path.splitext(base)
            dst = os.path.join(dest, f"{rank:04d}_{s:.3f}_{stem}__{n}{ext}"); n += 1
        if mode == "移动（带还原清单）":
            shutil.move(p, dst); rows.append([dst, p])
        else:
            shutil.copy2(p, dst)
        done += 1
    msg = f"已{('移动' if rows else '复制')} {done} 张到：{dest}"
    if rows:
        man = os.path.join(dest, "_还原清单.csv")
        with open(man, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f); w.writerow(["现位置", "原始位置"]); w.writerows(rows)
        msg += f"\n还原清单：{man}（运行 restore_models 风格脚本或手动按清单移回）"
    return msg

def _gallery(items, cap=300):
    """items: list of (path, score) -> gradio gallery list"""
    out = []
    for p, s in items[:cap]:
        out.append((p, f"{s:.3f}  {os.path.basename(p)}"))
    return out

# ================= 界面 =================
with gr.Blocks(title="照片识别系统") as demo:
    gr.Markdown("# 📷 照片识别系统（本地·免费）\n中文 CLIP + InsightFace，全部在你电脑本地运行。")

    # ---- Tab1 场景搜索 ----
    with gr.Tab("① 场景搜索"):
        gr.Markdown("按**内容**找图，例如：`工厂生产场景 车间 流水线` / `户外阳光` / `白底产品图`")
        s_dir = gr.Textbox(label="照片目录", value="/path/to/your/photos")
        s_q = gr.Textbox(label="描述（中文）", value="工厂 车间 生产线 流水线 工人作业 机器设备 生产现场")
        with gr.Row():
            s_th = gr.Slider(0.25, 0.50, value=0.365, step=0.005, label="相似度阈值（越高越严）")
            s_max = gr.Slider(10, 500, value=150, step=10, label="最多显示张数")
        s_btn = gr.Button("🔍 搜索", variant="primary")
        s_info = gr.Markdown()
        s_gal = gr.Gallery(label="结果", columns=6, height=520)
        s_state = gr.State([])
        with gr.Row():
            s_dest = gr.Textbox(label="导出到文件夹", value="/path/to/your/photos/场景结果")
            s_mode = gr.Radio(["复制（原图保留）", "移动（带还原清单）"], value="复制（原图保留）", label="方式")
        s_exp = gr.Button("📁 导出当前结果")
        s_expinfo = gr.Markdown()

        def do_scene(d, q, th, mx, progress=gr.Progress()):
            paths, feats = build_clip_index(d, progress)
            tf = clip_text_vec(q)
            sims = feats @ tf
            order = np.argsort(-sims)
            res = [(paths[i], float(sims[i])) for i in order if sims[i] >= th]
            info = f"**{len(res)} 张** 相似度≥{th:.3f}（共 {len(paths)} 张）。下面显示前 {min(len(res), int(mx))} 张。"
            return _gallery(res, int(mx)), res, info
        s_btn.click(do_scene, [s_dir, s_q, s_th, s_max], [s_gal, s_state, s_info])
        s_exp.click(export, [s_state, s_dest, s_mode], s_expinfo)

    # ---- Tab2 找特定人物 ----
    with gr.Tab("② 找特定人物"):
        gr.Markdown("上传 **1~3 张清晰正脸**作参考，在目录里找同一个人（换衣服/跨时间都行）。")
        p_dir = gr.Textbox(label="照片目录", value="/path/to/your/photos")
        p_refs = gr.File(label="参考正脸（可多张）", file_count="multiple", file_types=["image"])
        p_th = gr.Slider(0.20, 0.55, value=0.35, step=0.01, label="相似度阈值（0.3宽松 0.4严格）")
        p_max = gr.Slider(10, 500, value=150, step=10, label="最多显示张数")
        p_btn = gr.Button("🔍 查找此人", variant="primary")
        p_info = gr.Markdown()
        p_gal = gr.Gallery(label="结果", columns=6, height=520)
        p_state = gr.State([])
        with gr.Row():
            p_dest = gr.Textbox(label="导出到文件夹", value="/path/to/your/photos/某人")
            p_mode = gr.Radio(["复制（原图保留）", "移动（带还原清单）"], value="复制（原图保留）", label="方式")
        p_exp = gr.Button("📁 导出当前结果")
        p_expinfo = gr.Markdown()

        def do_person(d, refs, th, mx, progress=gr.Progress()):
            if not refs:
                raise gr.Error("请先上传至少 1 张参考正脸。")
            app = get_face()
            embs = []
            for rf in refs:
                img = load_rgb(rf.name if hasattr(rf, "name") else rf)
                if img is None: continue
                fs = app.get(_to_bgr(img))
                if fs:
                    big = max(fs, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
                    embs.append(big.normed_embedding)
            if not embs:
                raise gr.Error("参考图里没检测到人脸，请换更清晰的正脸。")
            ref = np.mean(embs, axis=0); ref = ref/np.linalg.norm(ref)
            paths, data = build_face_index(d, progress)
            res = []
            for i, p in enumerate(paths):
                e = data[f"e{i}"]
                if len(e) == 0: continue
                best = float(np.max(e @ ref))
                if best >= th: res.append((p, best))
            res.sort(key=lambda x: -x[1])
            info = f"**命中 {len(res)} 张**（阈值 {th:.2f}）。参考脸用了 {len(embs)} 张合成。"
            return _gallery(res, int(mx)), res, info
        p_btn.click(do_person, [p_dir, p_refs, p_th, p_max], [p_gal, p_state, p_info])
        p_exp.click(export, [p_state, p_dest, p_mode], p_expinfo)

    # ---- Tab3 人物照筛选 ----
    with gr.Tab("③ 人物照/模特图筛选"):
        gr.Markdown("挑出**有真人**的照片；可叠加 CLIP「模特穿戴产品」语义排序，过滤自拍/工人。")
        m_dir = gr.Textbox(label="照片目录", value="/path/to/your/photos")
        with gr.Row():
            m_conf = gr.Slider(0.5, 0.95, value=0.8, step=0.05, label="人脸置信度≥（踢掉头盔误检）")
            m_prom = gr.Slider(0.0, 0.05, value=0.005, step=0.001, label="人物占比≥（人脸/整图）")
        m_use_clip = gr.Checkbox(value=True, label="用 CLIP「模特穿戴产品」二次排序/过滤")
        m_clipq = gr.Textbox(label="模特语义描述", value="模特穿戴运动护具和球衣展示产品 一个人穿着冰球装备")
        m_clipth = gr.Slider(0.30, 0.50, value=0.38, step=0.005, label="模特语义分≥（勾选上面才生效）")
        m_max = gr.Slider(10, 500, value=150, step=10, label="最多显示张数")
        m_btn = gr.Button("🔍 筛选", variant="primary")
        m_info = gr.Markdown()
        m_gal = gr.Gallery(label="结果", columns=6, height=520)
        m_state = gr.State([])
        with gr.Row():
            m_dest = gr.Textbox(label="导出到文件夹", value="/path/to/your/photos/模特图")
            m_mode = gr.Radio(["复制（原图保留）", "移动（带还原清单）"], value="复制（原图保留）", label="方式")
        m_exp = gr.Button("📁 导出当前结果")
        m_expinfo = gr.Markdown()

        def do_models(d, conf, prom, use_clip, clipq, clipth, mx, progress=gr.Progress()):
            paths, data = build_face_index(d, progress)
            cand = []
            for i, p in enumerate(paths):
                m = data[f"m{i}"]
                if len(data[f"e{i}"]) == 0: continue
                pr, cf = float(m[0]), float(m[1])
                if cf >= conf and pr >= prom:
                    cand.append((p, pr, i))
            score_map = {}
            if use_clip:
                cpaths, feats = build_clip_index(d, progress)
                cidx = {pp: j for j, pp in enumerate(cpaths)}
                tf = clip_text_vec(clipq)
                kept = []
                for p, pr, i in cand:
                    if p in cidx:
                        sc = float(feats[cidx[p]] @ tf)
                        if sc >= clipth:
                            kept.append((p, sc))
                kept.sort(key=lambda x: -x[1])
                res = kept
                info = f"**{len(res)} 张**：有真人(conf≥{conf}, 占比≥{prom*100:.1f}%) 且 模特分≥{clipth}。"
            else:
                res = sorted([(p, pr) for p, pr, i in cand], key=lambda x: -x[1])
                info = f"**{len(res)} 张**：有真人(conf≥{conf}, 占比≥{prom*100:.1f}%)，按人物占比排序。"
            return _gallery(res, int(mx)), res, info
        m_btn.click(do_models, [m_dir, m_conf, m_prom, m_use_clip, m_clipq, m_clipth, m_max],
                    [m_gal, m_state, m_info])
        m_exp.click(export, [m_state, m_dest, m_mode], m_expinfo)

    gr.Markdown("提示：首次对某目录搜索会先建索引（较慢），之后走缓存秒出。模型与索引都在本地。")

if __name__ == "__main__":
    # allowed_paths 放行外置盘/用户目录，否则 Gradio 不显示这些位置的缩略图
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, inbrowser=False,
                        show_error=True, allowed_paths=["/Volumes", "/Users"])
