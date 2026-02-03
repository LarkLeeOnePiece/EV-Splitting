# EVSplitting - Interactive 3D Gaussian Splatting with Event-based Splitting

[English](README_EN.md) | [中文](README_CN.md)

<p align="center">
  <img src="resources/images/teaser.png" width="800" alt="EVSplitting Demo"/>
  <br>
  <em>Interactive 3D Gaussian Splatting with Event-based Splitting Visualization</em>
</p>

## About | 关于

**English:** An unofficial open-source implementation of Event-based Gaussian Splitting for 3D Gaussian Splatting, featuring an interactive visualization tool built on Splatviz.

**中文：** Event-based Gaussian Splitting 的非官方开源实现，基于 Splatviz 构建的交互式 3D Gaussian Splatting 可视化工具。

---

## 📄 Paper | 论文

**Paper Title (Placeholder):** [论文标题占位符]
- **Conference:** SIGGRAPH Asia 2024 / ACM Transactions on Graphics
- **DOI:** [10.1145/3680528.3687592](https://dl.acm.org/doi/full/10.1145/3680528.3687592)

## 🔗 Based on | 基于

- **[Splatviz](https://github.com/Florian-Barthel/splatviz)** - Interactive 3D Gaussian Splatting Viewer by Florian Barthel

---

## ✨ Key Features | 核心特性

- 🎯 **Event-based Gaussian Splitting** - CUDA加速的自适应分割算法
- 🎨 **Interactive GUI** - 基于ImGui的实时可视化
- 🧩 **Multi-plane Clipping** - 多平面裁剪与可视化
- 💾 **Memory Optimization** - 场景图高效内存管理
- 📊 **Benefit-Cost Control** - 代理控制的分割策略

---

## 🚀 Quick Start | 快速开始

```bash
# Install dependencies first
pip install torch torchvision imgui-bundle click numpy imageio loguru Pillow open3d

# Build CUDA extensions
cd gaussian-splatting/submodules/ev-splitting && pip install -e .
cd ../simple-knn && pip install -e .
cd ../../..

# Run the application
python run_main.py --data_path=/path/to/your/ply/files
```

---

## 📚 Documentation | 文档

Choose your preferred language to get started:

选择您偏好的语言开始使用：

### 📘 English Documentation
For detailed installation, usage, and algorithm information, please refer to **[README_EN.md](README_EN.md)**

- Installation Guide | 安装指南
- Quick Start Tutorial | 快速开始教程
- EVS Algorithm Explanation | 算法说明
- GUI Controls Reference | 界面控制说明
- Examples | 使用示例
- Citation | 引用信息

### 📗 中文文档
详细的安装、使用和算法说明，请参考 **[README_CN.md](README_CN.md)**

- 安装指南
- 快速开始教程
- EVS 算法说明
- 界面控制说明
- 使用示例
- 引用信息

---

## 🎮 Features Overview | 功能概览

### EVS Splitting Modes | 分割模式

| Mode | Description | Use Case |
|------|-------------|----------|
| **Naive** | Split all intersecting Gaussians | General purpose |
| **Proxy Control** | Benefit-cost based splitting | Quality optimization |

### Memory Optimization | 内存优化

| Mode | Memory Usage | Best For |
|------|------|----------|
| **Naive** | Higher | Small scenes |
| **Scene Graph** | Medium | Balanced |
| **CPU Offload** | Lowest | Large scenes |

---

## 🛠️ System Requirements | 系统要求

- **OS:** Windows / Linux / macOS (with CUDA support)
- **GPU:** NVIDIA (Compute Capability ≥ 7.0)
- **CUDA:** 11.0 or higher
- **Python:** 3.8 or higher
- **RAM:** 8GB minimum, 16GB recommended

---

## 🙏 Acknowledgements | 致谢

This project builds upon:

- **[Splatviz](https://github.com/Florian-Barthel/splatviz)** - Interactive viewer
- **[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)** - Original implementation by INRIA GRAPHDECO
- **[diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)** - Differentiable rendering core

---

## 📄 License | 许可证

This project is licensed for **non-commercial research and evaluation use only**.

本项目仅供**非商业研究和评估使用**。

For commercial licensing inquiries, please contact the original paper authors.

---

## 📧 Questions? | 有问题？

- **For implementation issues:** Open an issue on GitHub
- **For research questions:** Please refer to the paper
- **For Splatviz-related questions:** See [Splatviz Repository](https://github.com/Florian-Barthel/splatviz)

---

<p align="center">
  Made with ❤️ for the 3D Gaussian Splatting community
  <br>
  <br>
  ⭐ If you find this useful, please star the repository!
</p>
