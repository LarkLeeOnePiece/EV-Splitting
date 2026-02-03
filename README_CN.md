# EVSplitting - 交互式 3D 高斯点云可视化工具

[![许可证](https://img.shields.io/badge/许可证-非商业使用-blue.svg)](#-许可证)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](#-系统要求)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-orange.svg)](#-系统要求)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](#-安装)

[中文](README_CN.md) | **[English](README_EN.md)**

## 📖 项目简介

EVSplitting 是 Event-based Gaussian Splitting (EVS) 方法的**非官方**开源实现，用于 3D Gaussian Splatting 的交互式可视化。本项目基于 [Splatviz](https://github.com/Florian-Barthel/splatviz) 开发，提供以下功能：

- 🎯 **Event-based Gaussian 分割（EVS）** - CUDA加速的自适应高斯分割算法
- 🎨 **交互式图形界面** - 基于ImGui的实时可视化控制
- 🧩 **多平面裁剪支持** - 支持多平面裁剪与可视化
- 💾 **内存优化** - 基于场景图的高效内存管理（3种模式）
- 📊 **收益-代价控制** - 基于代理的分割策略，支持可配置代价函数
- 🎬 **媒体导出** - 视频录制和截图保存

---

## 🔗 相关链接

- 📄 **论文：** [论文标题占位符](https://dl.acm.org/doi/full/10.1145/3680528.3687592)
  - 会议：SIGGRAPH Asia 2024
  - DOI: [10.1145/3680528.3687592](https://dl.acm.org/doi/full/10.1145/3680528.3687592)

- 🏗️ **基于项目：** [Splatviz](https://github.com/Florian-Barthel/splatviz) 作者：Florian Barthel
  - 交互式 3D 高斯点云查看器
  - 可视化框架和渲染管道

---

## ✨ 核心特性

### EVS 分割模式

#### 1. **朴素模式** (`evs_split_mode=0`)
- 分割所有与裁剪平面相交的高斯
- 支持 1-5 次自适应迭代进行逐步优化
- 适用于通用分割场景

```python
# 示例配置
enable_evs = True
evs_split_mode = 0
evs_max_passes = 3  # 1-5 次迭代
```

#### 2. **代理控制模式** (`evs_split_mode=1`)
- 基于收益-代价分析的选择性分割
- 两种代价函数选择：
  - **非对称：** `1-min(Cl,Cr)` - 优先平衡左右颜色
  - **保守型：** `|Cl-Cr|` - 最小化差异
- 可配置 lambda 阈值以调节收益-代价权衡

```python
# 示例配置
evs_split_mode = 1
evs_cost_mode = 0    # 0: 非对称, 1: 保守型
evs_lambda = 1.0     # 收益-代价阈值
```

### 内存优化模式

项目提供三种内存优化策略：

| 模式 | 内存使用 | 最适用于 | 主要特性 |
|------|------|----------|-------------|
| **Naive（朴素）** | 克隆原始数据 | 小型场景 | 简单，内存需求高 |
| **Scene Graph（场景图）** | 基于引用 | 中等规模场景 | 平衡性能和内存 |
| **CPU Offload（CPU卸载）** | 最小化GPU占用 | 大型场景 | 最大化GPU内存节省 |

每种模式都在性能和内存效率之间提供不同的权衡。

### 交互式控制

- **实时参数调整** - 即时获得视觉反馈
- **裁剪平面编辑器** - 多平面编辑与可视化
- **性能监控** - 实时FPS和内存统计
- **视频录制** - 捕获旋转视频
- **截图保存** - 导出渲染图像

---

## 🛠️ 安装指南

### 前置要求

1. **系统要求：**
   - **操作系统：** Windows / Linux / macOS（需CUDA支持）
   - **GPU：** NVIDIA 显卡，计算能力 ≥ 7.0
   - **CUDA：** 11.0 或更高版本
   - **cuDNN：** 8.0 或更高版本（推荐）
   - **Python：** 3.8 或更高版本
   - **内存：** 最少 8GB，推荐 16GB

2. **验证 CUDA 安装：**
   ```bash
   nvidia-smi
   nvcc --version
   ```

### 第 1 步：安装 PyTorch 和依赖

```bash
# 安装 PyTorch（CUDA 11.8 版本）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install imgui-bundle click numpy imageio loguru Pillow open3d
```

**CUDA 12.1 版本替代方案：**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 第 2 步：编译 CUDA 扩展

进入项目根目录，按顺序编译 CUDA 扩展：

```bash
# 1. 编译 EV-Splitting 扩展（EVS 算法的 CUDA 核心）
cd gaussian-splatting/submodules/ev-splitting
pip install -e .

# 2. 编译 Simple-KNN 扩展（光栅化的K近邻）
cd ../simple-knn
pip install -e .

# 3. 返回项目根目录
cd ../../..
```

**注意：** CUDA 扩展编译可能需要 5-10 分钟。这将编译 CUDA 代码（`evs_split.cu`、`forward.cu`、`backward.cu`）为 Python 扩展。

### 第 3 步：验证安装

```bash
# 测试 EVSplitting 模块
python -c "import evsplitting; print('✅ EVSplitting 安装成功！')"

# 测试 Simple-KNN 模块
python -c "import simple_knn; print('✅ Simple-KNN 安装成功！')"
```

### 完整安装示例

```bash
# 克隆仓库
git clone <repository-url>
cd EVSplitting

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装依赖
pip install imgui-bundle click numpy imageio loguru Pillow open3d

# 编译 CUDA 扩展
cd gaussian-splatting/submodules/ev-splitting && pip install -e .
cd ../simple-knn && pip install -e .
cd ../../..

# 验证安装
python -c "import evsplitting; print('安装完成！')"
```

---

## 🚀 使用方法

### 快速开始

```bash
python run_main.py --data_path=/path/to/your/ply/files
```

### 命令行参数

```bash
python run_main.py \
  --data_path /path/to/scenes \  # 必需：包含 .ply 文件的目录
  --mode default \                # [default, decoder, attach]（默认：default）
  --host 127.0.0.1 \             # 服务器地址（默认：127.0.0.1）
  --port 6009                     # 服务器端口（默认：6009）
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|--------|------|---------|-------------|
| `--data_path` | 路径 | 必需 | 包含场景文件夹的根目录，文件夹中应包含 `.ply` 文件 |
| `--mode` | 字符串 | default | 渲染模式：`default`（标准）、`decoder`（带高斯解码器）、`attach`（远程连接） |
| `--host` | 字符串 | 127.0.0.1 | 可视化服务器的主机地址 |
| `--port` | 整数 | 6009 | 可视化服务器的端口号 |
| `--ggd_path` | 路径 | （可选） | Gaussian GAN Decoder 项目的路径（decoder 模式需要） |

### 使用示例

**示例 1：使用示例场景**
```bash
python run_main.py --data_path=./resources/sample_scenes/mytest
```

**示例 2：从自定义目录加载场景**
```bash
python run_main.py --data_path=D:\Projects\Datasets\MyScenes
```

**示例 3：在特定端口运行**
```bash
python run_main.py --data_path=/data/scenes --host 0.0.0.0 --port 8888
```

---

## 📊 数据集格式

EVSplitting 使用与 Splatviz 和标准 3D Gaussian Splatting 相同的数据集格式：

```
data_path/
├── scene1/
│   ├── point_cloud.ply         # 3D 高斯点云 PLY 文件（必需）
│   └── cameras.json            # 相机参数（可选）
├── scene2/
│   ├── point_cloud.ply
│   └── cameras.json
└── scene3/
    └── point_cloud.ply
```

### PLY 文件要求

`.ply` 文件必须包含高斯点云的以下属性：

```
# 必需属性：
- x, y, z                          # 3D 位置
- nx, ny, nz                       # 法向量（或辅助数据）
- red, green, blue                 # 颜色（或使用 SH 系数）
- opacity                          # 透明度
- scale_0, scale_1, scale_2       # 3D 尺度
- rot_0, rot_1, rot_2, rot_3      # 四元数旋转
- sh_0_0, sh_0_1, ... sh_3_15     # 球谐系数（可选）
```

### 兼容的数据源

- ✅ **官方 3D Gaussian Splatting** - 直接兼容
- ✅ **基于 COLMAP 的训练输出** - 完全支持
- ✅ **自定义实现** - 只要输出包含上述属性的 PLY 文件

---

## 📐 EVS 分割算法

### 算法概述

Event-based Gaussian Splitting 解决了从垂直于裁剪平面的角度查看 3D Gaussian Splatting 场景时的渲染伪影问题。算法自适应地分割与裁剪平面相交的高斯。

### 数学表述

#### 1. 朴素分割

对于每个裁剪平面，识别相交的高斯并分割它们：

```
对于每个高斯 G：
  如果 G 与裁剪平面相交：
    将 G 分割为 2 个子高斯
    递归应用于下一个平面（最多 max_passes 次）
```

#### 2. 收益-代价分析

对于每个候选高斯：

```
benefit = 分割带来的渲染质量改进
cost = Lambda * 新增高斯数量

如果 benefit > cost，则执行分割
```

**代价函数：**

- **非对称：** `cost = 1 - min(Cl, Cr)`
  - Cl, Cr：左右两侧的颜色覆盖
  - 优先平衡颜色分布

- **保守型：** `cost = |Cl - Cr|`
  - 最小化绝对差异
  - 更保守的分割策略

### 配置参数

```python
# EVS 核心参数
enable_evs = True                 # 启用 EVS 分割
evs_split_mode = 0               # 0=朴素, 1=代理控制
evs_max_passes = 2               # 分割迭代次数（1-5）
evs_min_split_threshold = 0      # 早停阈值

# 代理控制参数（仅当 evs_split_mode=1 时有效）
evs_cost_mode = 0                # 0=非对称, 1=保守型
evs_lambda = 1.0                 # 收益-代价权衡

# 内存优化
evs_mode = "naive"               # "naive"、"scenegraph"、"cpu_offload"
evs_measure_memory = False       # 启用内存分析
```

### 性能特性

| 配置 | 质量 | 速度 | 内存 |
|---------------|---------|-------|--------|
| 朴素 1-pass | 低 | 极快 | 高 |
| 朴素 3-pass | 中 | 快 | 高 |
| 代理控制 保守型 | 高 | 中等 | 高 |
| 场景图 + 代理控制 | 高 | 中等 | 中 |
| CPU卸载 + 代理控制 | 高 | 较慢 | 低 |

---

## 🎮 界面控件

交互式界面由多个控件面板组成：

### 主要控件

1. **Load（加载）** - 加载 PLY 或 PKL 文件
2. **Camera（相机）** - 相机控制和导航
   - 平移、缩放、旋转
   - 预设视角
3. **Render（渲染）** - 渲染设置
   - 背景颜色
   - 抗锯齿选项
4. **Edit（编辑）** - 高斯编辑工具
   - 笔刷式修改
5. **EVS Splitting（EVS 分割）**（核心控件）
   - 启用/禁用 EVS 分割
   - 配置分割模式（朴素/代理控制）
   - 设置最大迭代次数（1-5）
   - 选择代价函数
   - 设置 lambda 阈值
   - 选择内存优化模式
   - 启用内存测量
6. **Clipping Plane（裁剪平面）**
   - 添加/删除平面
   - 编辑平面法向量（nx, ny, nz）
   - 编辑平面距离（d）
   - 启用平面可视化
   - 剔除错误一侧的高斯
7. **Video（视频）** - 录制旋转视频
8. **Capture（捕获）** - 截图和图像导出
9. **Performance（性能）** - 实时统计信息
   - FPS 计数器
   - 内存使用量
   - 高斯数量

### 快捷键和鼠标操作

| 操作 | 控制方式 |
|--------|---------|
| 旋转视角 | 右键 + 拖动 |
| 平移视角 | 中键 + 拖动 |
| 缩放 | 鼠标滚轮 |
| 重置视角 | 空格键 |

---

## 🎨 教程示例

### 示例 1：基础 EVS 分割

```bash
# 1. 启动应用
python run_main.py --data_path=./resources/sample_scenes/mytest

# 2. 在界面中：
# - 点击"Load"并选择一个 .ply 文件
# - 导航到"EVS Splitting"控件
# - 勾选"Enable EVS Split"
# - 调整"EVS Passes"（从 1 开始，可增加到 3）
# - 观察裁剪平面边缘的渲染改进
```

### 示例 2：裁剪平面可视化

```bash
# 1. 启动应用并加载场景
python run_main.py --data_path=./your/scenes

# 2. 在界面中：
# - 导航到"Clipping Plane"控件
# - 勾选"Visualize Plane"
# - 调整平面法向量（nx, ny, nz）值
# - 调整平面距离（d）值
# - 平面交集多边形将在视口中出现

# 3. 在"EVS Splitting"控件中：
# - 勾选"Clip Model (Cull Gaussians)"
# - 错误一侧的高斯将被剔除
```

### 示例 3：大型场景的内存优化

```bash
# 对于包含数百万高斯的场景：

# 1. 启动应用
python run_main.py --data_path=/path/to/large/scenes

# 2. 在界面中：
# - 导航到"EVS Splitting"控件
# - 选择"Memory Mode"→"CPU-Offload (Max Save)"
# - 启用"Measure Memory"查看改进
# - 这将显著降低 GPU 内存占用

# 3. 为了保证交互性：
# - 从"EVS Passes"= 1 开始
# - 使用"Proxy Control"模式配合保守型代价函数
```

### 示例 4：批量处理多个场景

```bash
# 如果在一个目录中有多个场景：
python run_main.py --data_path=/path/to/multiple/scenes

# 在界面中，使用 Load 控件在场景之间切换
```

---

## 📊 性能优化

### 实时交互

- 从 1-2 次 EVS 迭代开始
- 使用朴素内存模式
- 启用 Debug Render 以可视化分割

```python
evs_max_passes = 1
evs_mode = "naive"
evs_debug = True
```

### 高质量渲染

- 使用 3-5 次 EVS 迭代
- 使用代理控制模式配合保守型代价函数
- 使用场景图内存模式

```python
evs_max_passes = 5
evs_split_mode = 1
evs_cost_mode = 1
evs_mode = "scenegraph"
```

### 大型场景（>100万高斯）

- 使用 CPU Offload 内存模式
- 减少最大迭代次数
- 启用早停阈值

```python
evs_max_passes = 2
evs_min_split_threshold = 100
evs_mode = "cpu_offload"
evs_measure_memory = True
```

### 调试技巧

- **启用 Debug Render** - 分割以红/绿色显示
- **启用内存测量** - 监控 GPU 使用情况
- **检查 FPS** - 实时性能反馈
- **调整 Lambda** - 寻找最优的收益-代价权衡

---

## 📚 高级特性

### 场景图内存优化

场景图模式使用引用而不是克隆：

```python
# 在界面中：选择"Memory Mode"→"SceneGraph (Efficient)"

# 优点：
# - 无需初始克隆所有高斯
# - 仅存储增量变化
# - 多次分割时高效

# 适用场景：中等到大型场景的多次分割迭代
```

### 收益-代价控制

微调质量和内存之间的权衡：

```python
# 示例 1：激进分割（优先质量）
evs_lambda = 0.5      # 较低阈值 = 更多分割

# 示例 2：保守分割（优先内存）
evs_lambda = 2.0      # 较高阈值 = 较少分割

# 根据具体场景调整以找到最优平衡
```

### 多平面裁剪

支持同时使用多个裁剪平面：

```bash
# 在界面中：
# 1. 导航到"Clipping Plane"控件
# 2. 多次点击"Add Plane"
# 3. 分别配置每个平面
# 4. 场景将被所有平面同时裁剪
```

---

## 🐛 故障排除

### 常见问题

**问题：ModuleNotFoundError: No module named 'evsplitting'**
```bash
# 解决方案：重新编译 CUDA 扩展
cd gaussian-splatting/submodules/ev-splitting
pip install -e .
cd ../../..
```

**问题：CUDA 内存溢出**
```bash
# 解决方案：
# 1. 使用 Scene Graph 或 CPU Offload 内存模式
# 2. 减少 EVS 迭代次数（evs_max_passes = 1）
# 3. 降低图像分辨率
# 4. 使用更小的场景
```

**问题：渲染性能缓慢**
```bash
# 解决方案：
# 1. 从 evs_max_passes = 1 开始
# 2. 禁用 Debug Render（如果启用）
# 3. 禁用内存测量（如果启用）
# 4. 使用朴素内存模式以获得速度
```

**问题：裁剪平面不可见**
```bash
# 解决方案：
# 1. 检查"Clipping Plane"控件中的"Visualize Plane"
# 2. 确保平面与高斯相交
# 3. 检查平面法向量（应该是归一化的）
```

---

## 📝 引用

如果您在研究中使用此代码，请引用原始论文：

```bibtex
@article{evs_placeholder_2024,
  title={[论文标题占位符]},
  author={[作者占位符]},
  journal={ACM Transactions on Graphics (TOG)},
  volume={43},
  number={6},
  year={2024},
  publisher={ACM},
  doi={10.1145/3680528.3687592}
}
```

同时引用相关工作：

```bibtex
@article{kerbl2023gaussian,
  title={3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  author={Kerbl, Bernhard and others},
  journal={ACM Transactions on Graphics (TOG)},
  year={2023}
}

@inproceedings{barthel2024splatviz,
  title={Splatviz: Interactive Visualization for 3D Gaussian Splatting},
  author={Barthel, Florian and others},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2024}
}
```

---

## 🙏 致谢

本项目得益于以下优秀的开源工作：

### 主要参考

- **[Splatviz](https://github.com/Florian-Barthel/splatviz)** 作者：Florian Barthel
  - 交互式 3D 高斯点云查看器
  - GUI 框架和渲染管道

- **[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)** 作者：INRIA GRAPHDECO
  - 原始 3D Gaussian Splatting 实现
  - 核心高斯表示和光栅化

- **[diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)**
  - 可微分高斯光栅化
  - 基于 CUDA 的渲染后端

### 所使用的技术

- **PyTorch** - 深度学习框架
- **CUDA/cuDNN** - GPU 加速
- **ImGui** - 即时模式 GUI
- **OpenGL** - 图形渲染
- **GLM** - 数学库

### 特别感谢

特别感谢原始 EVS 论文的作者们的开创性研究，以及 3D Gaussian Splatting 社区的持续支持。

---

## 📄 许可证

本项目仅供**非商业研究和评估使用**，遵循 INRIA Gaussian Splatting 许可证条款。

### 使用权限

✅ **允许：**
- 学术研究和出版
- 教育用途
- 非商业评估和对比
- 衍生研究工作（需标注出处）

❌ **不允许：**
- 商业应用
- 商业许可或转售
- 未经明确许可将其纳入专有产品

### 商业用途

如需商业许可，请联系原论文作者。

---

## 🐛 问题反馈与贡献

### 报告问题

- 请在 GitHub Issues 中报告 bug
- 包含以下信息：
  - 错误信息和堆栈跟踪
  - 系统信息（操作系统、GPU、CUDA 版本、Python 版本）
  - 重现步骤
  - 截图或日志（如适用）

### 代码贡献

欢迎贡献！请：

1. Fork 本仓库
2. 创建功能分支（`git checkout -b feature/improvement`）
3. 提交更改
4. 使用清晰的说明提交 Pull Request

### 研究问题

关于原始 EVS 算法或论文的问题：
- 参考原始论文 [10.1145/3680528.3687592](https://dl.acm.org/doi/full/10.1145/3680528.3687592)
- 联系原论文作者

---

## 📧 联系与支持

- **实现问题：** 在 GitHub 上提交 Issue
- **功能请求：** GitHub Discussions 或 Issues
- **研究合作：** 联系原论文作者
- **Splatviz 问题：** 查看 [Splatviz 仓库](https://github.com/Florian-Barthel/splatviz)

---

## 📖 额外资源

- **3D Gaussian Splatting 论文：** [Kerbl et al., 2023](https://repo.cvg.ethz.ch/projects/3dgs/)
- **Splatviz 文档：** [GitHub Wiki](https://github.com/Florian-Barthel/splatviz/wiki)
- **CUDA 编程指南：** [NVIDIA 文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

---

<p align="center">
  <strong>❤️ 为 3D Gaussian Splatting 社区开发</strong>
  <br>
  <br>
  <a href="https://github.com">⭐ 如果你觉得有帮助，请给仓库点个星！</a>
</p>
