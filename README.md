# DMORE 智能脸谱 · 本地识图工作站

**中文** | [English](./README_EN.md)

基于 **中文 CLIP + InsightFace + ffmpeg** 的本地图片/视频识别软件，全部在本机运行，免费、不联网、照片不外传。

## 启动
- **桌面应用**：双击 `DMORE智能脸谱.app`（原生窗口，自动启动服务）
- **或网页版**：双击 `启动DMORE智能脸谱.command`，浏览器开 http://127.0.0.1:8800
- 命令行：`./venv/bin/python -m uvicorn server.main:app --port 8800`

## 功能（15 模块）
| 分类 | 模块 |
|------|------|
| 发现 | 仪表盘（统计+内容分布+人物概览） |
| 搜索 | 语义搜索（文字找图）、以图搜图 |
| 人物 | 人物图谱（自动聚类）、找某个人（参考脸）、同框关系 |
| 整理 | 智能标签、自动相册、去重清理、质量筛选 |
| 行业 | 电商选图台（产品/模特/场景/工厂分流） |
| 视频 | 视频检索（按内容/定位时间点）、分镜抽帧、片段导出 |
| 操作 | 批量处理、图库与索引、设置 |

## 安全
- **只移动不删除**：所有「移动/去重清理」都会在目标文件夹写 `_还原清单.csv`，可一键还原；永不删除原图。

## 结构
```
server/core.py   引擎与索引（CLIP/人脸/视频抽帧/质量/镜头）
server/main.py   FastAPI 接口
web/             前端（index.html + app.js + 样式 + 资源）
desktop.py       桌面应用入口（pywebview 原生窗口）
cache/           索引缓存（按目录）   models/  模型权重
```

## 模型 / 镜像
- CLIP：`cn_clip ViT-B-16`（魔搭 ModelScope 下载）
- 人脸：`InsightFace buffalo_l`
- pip 走阿里云镜像；HuggingFace 在国内不可用，已改魔搭。

## 版权与开源协议 / License

**Copyright © 2026 DMORE / Renmingsen**

本项目以 **GNU General Public License v3.0 (GPL-3.0-or-later)** 授权开源，
完整条款见 [`LICENSE`](./LICENSE)。你可自由使用、修改、分发，
但**分发修改版须同样以 GPL 开源并附源码**。

- 版权与第三方组件声明见 [`NOTICE.md`](./NOTICE.md)。
- ⚠️ **InsightFace 预训练模型仅限非商业研究用途**；商业使用需另行获取模型授权。
- DMORE / D·MORE 名称及 Logo 为品牌标识，不在 GPL 授权范围内。
