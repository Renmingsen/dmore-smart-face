# 版权与第三方声明 / Copyright & Third-Party Notices

## 版权所属 / Copyright

DMORE 智能脸谱（DMORE Vision）— 本地识图工作站

**Copyright © 2026 DMORE / Renmingsen（https://github.com/Renmingsen）**
保留所有权利（在 GPL-3.0 授权范围内）。

本项目源代码采用 **GNU General Public License v3.0 (GPL-3.0-or-later)** 授权，
完整条款见同目录 `LICENSE` 文件，或 <https://www.gnu.org/licenses/gpl-3.0.html>。

> 简言之（不构成法律意见）：你可以自由使用、修改、分发本项目，
> 但**分发修改版时必须同样以 GPL 开源并附带源码**。

DMORE / D·MORE 名称与 Logo 为品牌标识，版权归 DMORE 所有；
GPL 授权**不含**对品牌商标的使用许可。

---

## 第三方组件 / Third-Party Components

本项目调用以下开源组件，各自遵循其原始许可证：

| 组件 | 用途 | 许可证 |
|------|------|--------|
| Chinese-CLIP (cn_clip) | 中文图文语义 | MIT |
| PyTorch / torchvision | 深度学习框架 | BSD-3 |
| InsightFace | 人脸检测/识别 | MIT（代码）|
| ONNX Runtime | 人脸模型推理 | MIT |
| FastAPI / Starlette / Uvicorn | Web 后端 | MIT / BSD |
| Gradio（旧版界面） | 网页界面 | Apache-2.0 |
| pywebview | 桌面原生窗口 | BSD-3 |
| OpenCV (opencv-python) | 视频/图像处理 | Apache-2.0 |
| Pillow / pillow-heif | 图像读写/HEIC | MIT-CMU / BSD |
| NumPy | 数值计算 | BSD-3 |
| FFmpeg（外部命令调用） | 视频抽帧/裁剪 | LGPL/GPL（取决于构建）|

### ⚠️ 重要：模型权重的使用限制

- **InsightFace 预训练模型（buffalo_l 等）**：官方授权为
  **仅限非商业研究用途（non-commercial research only）**。
  如需**商业使用**，请自行联系 InsightFace 获取授权或改用可商用模型。
- **Chinese-CLIP / ViT-B-16 权重**：遵循其各自模型许可证。

模型权重**不随本仓库分发**（首次运行时从 ModelScope 自动下载）。
使用者需自行确认所用模型权重的授权是否符合自身用途。

### FFmpeg 说明
本项目通过 `subprocess` 调用系统已安装的 **ffmpeg 命令行**（未静态链接其代码），
因此不构成对本项目源码的许可证传染；ffmpeg 自身遵循其发行版的 LGPL/GPL 条款。
