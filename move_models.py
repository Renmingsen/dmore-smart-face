import json, os, csv, shutil

THRESH = 0.38
SRC_LIST = "cache/model_ranked.json"   # [score, prom, path]
DEST = "/path/to/your/photos/模特图"
MANIFEST = os.path.join(DEST, "_还原清单.csv")

cand = json.load(open(SRC_LIST, encoding="utf-8"))
picks = [(s, pr, p) for s, pr, p in cand if s >= THRESH]
picks.sort(reverse=True)
os.makedirs(DEST, exist_ok=True)

moved, missing, rows = 0, 0, []
for rank, (s, pr, src) in enumerate(picks, 1):
    if not os.path.isfile(src):
        missing += 1
        print("  [缺失,跳过]", src); continue
    base = os.path.basename(src)
    dst = os.path.join(DEST, f"{rank:03d}_{s:.3f}_{base}")
    # 防重名
    n = 1
    while os.path.exists(dst):
        stem, ext = os.path.splitext(base)
        dst = os.path.join(DEST, f"{rank:03d}_{s:.3f}_{stem}__{n}{ext}"); n += 1
    shutil.move(src, dst)            # 移动，不删除
    rows.append([dst, src, f"{s:.4f}", f"{pr:.4f}"])
    moved += 1

with open(MANIFEST, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["现位置", "原始位置", "模特分", "人物占比"])
    w.writerows(rows)

print(f"\n已移动 {moved} 张到: {DEST}")
print(f"缺失跳过: {missing}")
print(f"还原清单: {MANIFEST}（{len(rows)} 条）")
