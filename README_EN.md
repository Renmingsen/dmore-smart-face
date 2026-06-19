# DMORE Vision · Local Image-Recognition Studio

[中文](./README.md) | **English**

A **fully local** photo/video recognition app built on **Chinese-CLIP + InsightFace + ffmpeg**.
Everything runs on your own machine — free, offline, your photos never leave your computer.

> UI is bilingual: click the **中 / EN** button in the top bar to switch.

## Launch
- **Desktop app**: double-click `DMORE智能脸谱.app` (native window, auto-starts the local server)
- **Or web**: run `./venv/bin/python -m uvicorn server.main:app --port 8800`, then open http://127.0.0.1:8800
- The first run downloads models (~0.7 GB) from ModelScope automatically.

## Features (15 modules)
| Group | Modules |
|-------|---------|
| Discover | Dashboard (stats + content distribution + people) |
| Search | Semantic Search (text→image), Image Search |
| People | People Atlas (auto-clustering), Find a Person (by reference face), Co-occurrence |
| Organize | Smart Tags, Auto Albums, Dedupe, Quality Filter |
| Industry | E-commerce Picker (product / model / scene / factory split) |
| Video | Video Search (by content, jump to timestamp), Shots & Frames, Clip Export |
| Operations | Batch, Library & Index, Settings |

## Safety
- **Move, never delete**: every "move / dedupe cleanup" writes a `_还原清单.csv` (restore manifest)
  into the target folder so it is fully reversible. Originals are never deleted.

## Power saving
- Settings → **Power-saving** limits compute threads to about half the CPU cores, so indexing /
  detection won't max out your machine while you do other things.

## Layout
```
server/core.py   engines & indexing (CLIP / face / video keyframes / quality / shots)
server/main.py   FastAPI endpoints
web/             frontend (index.html + app.js + i18n.js + assets)
desktop.py       desktop entry (pywebview native window)
cache/           per-folder index cache    models/  model weights
```

## License
**Copyright © 2026 DMORE / Renmingsen**

Licensed under **GNU General Public License v3.0 (GPL-3.0-or-later)** — see [`LICENSE`](./LICENSE).
You may use, modify and distribute it freely, but **distributed modifications must also be GPL,
with source included**.

- Copyright & third-party notices: [`NOTICE.md`](./NOTICE.md)
- ⚠️ **InsightFace pretrained models are for non-commercial research only**; obtain a separate
  license (or use a commercial-friendly model) for commercial use.
- "DMORE / D·MORE" name and logo are trademarks, **not** covered by the GPL grant.
