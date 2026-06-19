# DMORE 智能脸谱 (DMORE Vision) — 本地识图工作站
# Copyright (C) 2026 DMORE / Renmingsen
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See <https://www.gnu.org/licenses/>.

"""一键还原：把 模特图 文件夹里的图按清单移回各自原始位置。不删除任何文件。"""
import csv, os, shutil

MANIFEST = "/path/to/your/photos/模特图/_还原清单.csv"

rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
back, skip = 0, 0
for r in rows:
    cur, orig = r["现位置"], r["原始位置"]
    if not os.path.isfile(cur):
        print("  [当前文件不在,跳过]", cur); skip += 1; continue
    os.makedirs(os.path.dirname(orig), exist_ok=True)
    if os.path.exists(orig):
        print("  [原位置已存在,跳过]", orig); skip += 1; continue
    shutil.move(cur, orig); back += 1
print(f"已还原 {back} 张回原文件夹，跳过 {skip} 张。")
