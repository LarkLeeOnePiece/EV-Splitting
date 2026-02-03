# EVSplitting : An Efficient and Visually Consistent Splitting Algorithm for 3D Gaussian Splatting [paper](https://dl.acm.org/doi/full/10.1145/3680528.3687592)

<p align="center">
  <img src="resources/images/evs_demo.gif" width="800" alt="EVSplitting Demo"/>
  <br>
  <em>EV-Splitting Demo Visualization</em>
</p>

## About | 关于

**English:** An unofficial open-source implementation of Efficient and Visually Consistent Gaussian Splitting for 3D Gaussian Splatting, featuring an interactive visualization tool built on Splatviz.

**中文：** Efficient and Visually Consistent Gaussian Splitting 的非官方开源实现，基于 Splatviz 构建的交互式 3D Gaussian Splatting 可视化工具。

---

## 📄 Related Paper | 相关论文

**RaRa Clipper: A Clipper for Gaussian Splatting Based on Ray Tracer and Rasterizer:** 
- **Conference:** SIGGRAPH Asia 2025 / ACM Transactions on Graphics
- **DOI:** [10.1145/3757377.3763982](https://dl.acm.org/doi/full/10.1145/3757377.3763982)

## 🔗 Based on | 基于

- **[Splatviz](https://github.com/Florian-Barthel/splatviz)** - Interactive 3D Gaussian Splatting Viewer by Florian Barthel
- **[RaRaClipper](https://github.com/LarkLeeOnePiece/Openbase-RaRaClipper)** - Ray-Rasterization-based method for Gaussian Clipping
---

## ✨ Key Features | 核心特性

- 🎯 **Efficient and Visually Consistent Gaussian Splitting** - CUDA加速的自适应分割算法
- 🎨 **Interactive GUI** - 基于ImGui的实时可视化
- 🧩 **plane Clipping** - 平面裁剪与可视化
- 💾 **Memory Optimization** - 场景图高效内存管理
- 📊 **Benefit-Cost Control** - 代理控制的分割策略

---

## 🚀 Quick Start | 快速开始

```bash
# Install dependencies first
The envorinment is similar to splatviz. Please follow their instruction.

# Build CUDA extensions
cd gaussian-splatting/submodules/ev-splitting && pip install -e .


# Run the application
python run_main.py --data_path=/path/to/your/ply/files
```


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
- **[RaRaClipper](https://github.com/LarkLeeOnePiece/Openbase-RaRaClipper)** - RaRaClipper core

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
