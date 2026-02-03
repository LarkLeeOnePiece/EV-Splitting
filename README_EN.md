# EVSplitting - Interactive 3D Gaussian Splatting Viewer

[![License](https://img.shields.io/badge/License-Non--commercial-blue.svg)](#-license)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](#-system-requirements)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-orange.svg)](#-system-requirements)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](#-installation)

[中文文档](README_CN.md) | **English**

## 📖 Overview

EVSplitting is an **unofficial** open-source implementation of the Event-based Gaussian Splitting (EVS) method for 3D Gaussian Splatting. This project provides an interactive visualization tool built on top of [Splatviz](https://github.com/Florian-Barthel/splatviz), featuring:

- 🎯 **Event-based Gaussian Splitting (EVS)** - CUDA-accelerated adaptive splitting algorithm
- 🎨 **Interactive GUI** - Real-time visualization with ImGui-based controls
- 🧩 **Clipping Plane Support** - Multi-plane clipping with interactive visualization
- 💾 **Memory Optimization** - Scene graph-based efficient memory management (3 modes)
- 📊 **Benefit-Cost Control** - Proxy-based splitting strategy with configurable cost functions
- 🎬 **Media Export** - Video recording and screenshot capture

---

## 🔗 Related Links

- 📄 **Paper:** [Paper Title (Placeholder)](https://dl.acm.org/doi/full/10.1145/3680528.3687592)
  - Conference: SIGGRAPH Asia 2024
  - DOI: [10.1145/3680528.3687592](https://dl.acm.org/doi/full/10.1145/3680528.3687592)

- 🏗️ **Based on:** [Splatviz](https://github.com/Florian-Barthel/splatviz) by Florian Barthel
  - Original 3D Gaussian Splatting viewer
  - Interactive visualization framework

---

## ✨ Key Features

### EVS Splitting Modes

#### 1. **Naive Mode** (`evs_split_mode=0`)
- Splits all Gaussians intersecting with clipping planes
- Supports 1-5 adaptive passes for progressive refinement
- Best for general-purpose splitting

```python
# Example configuration
enable_evs = True
evs_split_mode = 0
evs_max_passes = 3  # 1-5 passes
```

#### 2. **Proxy Control Mode** (`evs_split_mode=1`)
- Benefit-cost analysis-based selective splitting
- Two cost function options:
  - **Asymmetric:** `1-min(Cl,Cr)` - Prioritizes left/right balance
  - **Conservative:** `|Cl-Cr|` - Minimize difference
- Configurable lambda threshold for cost-benefit tradeoff

```python
# Example configuration
evs_split_mode = 1
evs_cost_mode = 0    # 0: Asymmetric, 1: Conservative
evs_lambda = 1.0     # Benefit-cost threshold
```

### Memory Optimization Modes

The project offers three memory optimization strategies:

| Mode | Memory Usage | Best For | Key Feature |
|------|------|----------|-------------|
| **Naive** | Clone original data | Small scenes | Simple, highest memory |
| **Scene Graph** | Reference-based | Medium scenes | Balanced performance/memory |
| **CPU Offload** | Minimize GPU usage | Large scenes | Maximum GPU memory savings |

Each mode provides different tradeoffs between performance and memory efficiency.

### Interactive Controls

- **Real-time Parameter Adjustment** - Instant visual feedback
- **Clipping Plane Editor** - Multi-plane editing with visualization
- **Performance Monitoring** - Real-time FPS and memory statistics
- **Video Recording** - Capture rotation videos
- **Screenshot Capture** - Save rendered images

---

## 🛠️ Installation

### Prerequisites

1. **System Requirements:**
   - **OS:** Windows / Linux / macOS (with CUDA support)
   - **GPU:** NVIDIA GPU with CUDA support (Compute Capability ≥ 7.0)
   - **CUDA:** 11.0 or higher
   - **cuDNN:** 8.0 or higher (recommended)
   - **Python:** 3.8 or higher
   - **RAM:** 8GB minimum, 16GB recommended

2. **Verify CUDA Installation:**
   ```bash
   nvidia-smi
   nvcc --version
   ```

### Step 1: Install PyTorch and Dependencies

```bash
# Install PyTorch (CUDA 11.8 version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install imgui-bundle click numpy imageio loguru Pillow open3d
```

**Alternative for CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 2: Build CUDA Extensions

Navigate to the project root and build the CUDA extensions in order:

```bash
# 1. Build EV-Splitting extension (CUDA core for EVS algorithm)
cd gaussian-splatting/submodules/ev-splitting
pip install -e .

# 2. Build Simple-KNN extension (K-nearest neighbor for rasterization)
cd ../simple-knn
pip install -e .

# 3. Return to project root
cd ../../..
```

**Note:** Building CUDA extensions may take 5-10 minutes. This compiles the CUDA code (`evs_split.cu`, `forward.cu`, `backward.cu`) to Python extensions.

### Step 3: Verify Installation

```bash
# Test EVSplitting module
python -c "import evsplitting; print('✅ EVSplitting installed successfully!')"

# Test Simple-KNN module
python -c "import simple_knn; print('✅ Simple-KNN installed successfully!')"
```

### Complete Installation Example

```bash
# Clone the repository
git clone <repository-url>
cd EVSplitting

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install imgui-bundle click numpy imageio loguru Pillow open3d

# Build CUDA extensions
cd gaussian-splatting/submodules/ev-splitting && pip install -e .
cd ../simple-knn && pip install -e .
cd ../../..

# Verify
python -c "import evsplitting; print('Installation complete!')"
```

---

## 🚀 Usage

### Quick Start

```bash
python run_main.py --data_path=/path/to/your/ply/files
```

### Command Line Options

```bash
python run_main.py \
  --data_path /path/to/scenes \  # Required: Directory containing .ply files
  --mode default \                # [default, decoder, attach] (default: default)
  --host 127.0.0.1 \             # Server host (default: 127.0.0.1)
  --port 6009                     # Server port (default: 6009)
```

**Options Explanation:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--data_path` | Path | Required | Root directory containing scene folders with `.ply` files |
| `--mode` | String | default | Rendering mode: `default` (standard), `decoder` (with Gaussian decoder), `attach` (remote connection) |
| `--host` | String | 127.0.0.1 | Host address for the visualization server |
| `--port` | Integer | 6009 | Port number for the visualization server |
| `--ggd_path` | Path | (optional) | Path to Gaussian GAN Decoder project (required for decoder mode) |

### Examples

**Example 1: Basic usage with sample scenes**
```bash
python run_main.py --data_path=./resources/sample_scenes/mytest
```

**Example 2: Load scenes from custom directory**
```bash
python run_main.py --data_path=D:\Projects\Datasets\MyScenes
```

**Example 3: Run on specific port**
```bash
python run_main.py --data_path=/data/scenes --host 0.0.0.0 --port 8888
```

---

## 📊 Dataset Format

EVSplitting uses the same dataset format as Splatviz and standard 3D Gaussian Splatting:

```
data_path/
├── scene1/
│   ├── point_cloud.ply         # 3D Gaussian Splatting PLY file (required)
│   └── cameras.json            # Camera parameters (optional)
├── scene2/
│   ├── point_cloud.ply
│   └── cameras.json
└── scene3/
    └── point_cloud.ply
```

### PLY File Requirements

The `.ply` file must contain Gaussian point cloud properties:

```
# Required properties:
- x, y, z                          # 3D position
- nx, ny, nz                       # Normal vector (or auxiliary data)
- red, green, blue                 # Color (or use SH coefficients)
- opacity                          # Alpha transparency
- scale_0, scale_1, scale_2       # 3D scale
- rot_0, rot_1, rot_2, rot_3      # Quaternion rotation
- sh_0_0, sh_0_1, ... sh_3_15     # Spherical harmonics coefficients (optional)
```

### Compatible Sources

- ✅ **Official 3D Gaussian Splatting** - directly compatible
- ✅ **COLMAP-based training outputs** - fully supported
- ✅ **Custom implementations** - as long as they export PLY with the above properties

---

## 📐 EVS Splitting Algorithm

### Algorithm Overview

Event-based Gaussian Splitting addresses rendering artifacts when viewing 3D Gaussian Splatting scenes from angles perpendicular to clipping planes. The algorithm adaptively splits Gaussians that intersect with clipping planes.

### Mathematical Formulation

#### 1. Naive Splitting

For each clipping plane, identify intersecting Gaussians and split them:

```
For each Gaussian G:
  If G intersects clipping plane:
    Split G into 2 child Gaussians
    Recursively apply for next plane (up to max_passes)
```

#### 2. Benefit-Cost Analysis

For each candidate Gaussian:

```
benefit = Rendering quality improvement from splitting
cost = Lambda * Number of new Gaussians

Split if: benefit > cost
```

**Cost Functions:**

- **Asymmetric:** `cost = 1 - min(Cl, Cr)`
  - Cl, Cr: Color coverage on left/right sides
  - Prioritizes balanced color distribution

- **Conservative:** `cost = |Cl - Cr|`
  - Minimizes absolute difference
  - More selective splitting

### Configuration Parameters

```python
# Core EVS parameters
enable_evs = True                 # Enable EVS splitting
evs_split_mode = 0               # 0=Naive, 1=ProxyControl
evs_max_passes = 2               # Number of splitting iterations (1-5)
evs_min_split_threshold = 0      # Early termination threshold

# Proxy control parameters (evs_split_mode=1 only)
evs_cost_mode = 0                # 0=Asymmetric, 1=Conservative
evs_lambda = 1.0                 # Benefit-cost tradeoff

# Memory optimization
evs_mode = "naive"               # "naive", "scenegraph", "cpu_offload"
evs_measure_memory = False       # Enable memory profiling
```

### Performance Characteristics

| Configuration | Quality | Speed | Memory |
|---------------|---------|-------|--------|
| Naive 1-pass | Low | Very Fast | High |
| Naive 3-pass | Medium | Fast | High |
| ProxyControl Conservative | High | Medium | High |
| SceneGraph + ProxyControl | High | Medium | Medium |
| CPUOffload + ProxyControl | High | Slow | Low |

---

## 🎮 GUI Controls

The interactive interface consists of several widget panels:

### Main Widgets

1. **Load** - Load PLY or PKL files
2. **Camera** - Camera control and navigation
   - Pan, zoom, rotate
   - Preset views
3. **Render** - Rendering settings
   - Background color
   - Anti-aliasing options
4. **Edit** - Gaussian editing tools
   - Brush-based modifications
5. **EVS Splitting** (Core Widget)
   - Enable/disable EVS splitting
   - Configure split mode (Naive/Proxy Control)
   - Set maximum passes (1-5)
   - Choose cost function
   - Set lambda threshold
   - Select memory optimization mode
   - Enable memory measurement
6. **Clipping Plane**
   - Add/remove planes
   - Edit plane normal (nx, ny, nz)
   - Edit plane distance (d)
   - Enable plane visualization
   - Cull Gaussians on wrong side
7. **Video** - Record rotation videos
8. **Capture** - Screenshot and image export
9. **Performance** - Real-time statistics
   - FPS counter
   - Memory usage
   - Gaussian count

### Keyboard & Mouse Shortcuts

| Action | Control |
|--------|---------|
| Rotate view | Right mouse + drag |
| Pan view | Middle mouse + drag |
| Zoom | Mouse wheel |
| Reset view | Space |

---

## 🎨 Tutorial Examples

### Example 1: Basic EVS Splitting

```bash
# 1. Start the application
python run_main.py --data_path=./resources/sample_scenes/mytest

# 2. In the GUI:
# - Click "Load" and select a .ply file
# - Navigate to "EVS Splitting" widget
# - Check "Enable EVS Split"
# - Adjust "EVS Passes" (start with 1, increase to 3)
# - Observe the rendering improvement at clipping plane edges
```

### Example 2: Clipping Plane Visualization

```bash
# 1. Start with a scene loaded
python run_main.py --data_path=./your/scenes

# 2. In the GUI:
# - Navigate to "Clipping Plane" widget
# - Check "Visualize Plane"
# - Adjust plane normal (nx, ny, nz) values
# - Adjust plane distance (d) value
# - The plane intersection polygon will appear in the viewport

# 3. In "EVS Splitting" widget:
# - Check "Clip Model (Cull Gaussians)"
# - Gaussians on the wrong side will be culled
```

### Example 3: Memory Optimization for Large Scenes

```bash
# For scenes with millions of Gaussians:

# 1. Start application
python run_main.py --data_path=/path/to/large/scenes

# 2. In the GUI:
# - Navigate to "EVS Splitting" widget
# - Select "Memory Mode" → "CPU-Offload (Max Save)"
# - Enable "Measure Memory" to see improvements
# - This reduces GPU memory usage significantly

# 3. For interactive performance:
# - Start with "EVS Passes" = 1
# - Use "Proxy Control" mode with conservative cost
```

### Example 4: Batch Processing Multiple Scenes

```bash
# If you have multiple scenes in one directory:
python run_main.py --data_path=/path/to/multiple/scenes

# In GUI, use the Load widget to switch between scenes
```

---

## 📊 Performance Optimization

### For Real-Time Interaction

- Start with 1-2 EVS passes
- Use naive memory mode
- Enable Debug Render to visualize splits

```python
evs_max_passes = 1
evs_mode = "naive"
evs_debug = True
```

### For High-Quality Rendering

- Use 3-5 EVS passes
- Use Proxy Control mode with conservative cost
- Use Scene Graph memory mode

```python
evs_max_passes = 5
evs_split_mode = 1
evs_cost_mode = 1
evs_mode = "scenegraph"
```

### For Large Scenes (>1M Gaussians)

- Use CPU Offload memory mode
- Reduce maximum passes
- Enable early termination threshold

```python
evs_max_passes = 2
evs_min_split_threshold = 100
evs_mode = "cpu_offload"
evs_measure_memory = True
```

### Debugging Tips

- **Enable Debug Render** - Splits are shown in red/green
- **Enable Memory Measurement** - Monitor GPU usage
- **Check FPS** - Real-time performance feedback
- **Adjust Lambda** - Find optimal cost-benefit tradeoff

---

## 📚 Advanced Features

### Scene Graph Memory Optimization

The Scene Graph mode uses references instead of cloning:

```python
# In GUI: Select "Memory Mode" → "SceneGraph (Efficient)"

# Benefits:
# - No initial clone of all Gaussians
# - Only stores incremental changes
# - Efficient for multi-pass splitting

# Use case: Medium to large scenes with multiple splitting passes
```

### Benefit-Cost Control

Fine-tune the tradeoff between rendering quality and memory:

```python
# Example 1: Aggressive splitting (quality prioritized)
evs_lambda = 0.5      # Lower threshold = more splitting

# Example 2: Conservative splitting (memory prioritized)
evs_lambda = 2.0      # Higher threshold = less splitting

# Experiment to find optimal balance for your use case
```

### Multi-Plane Clipping

Support for multiple clipping planes simultaneously:

```bash
# In GUI:
# 1. Navigate to "Clipping Plane" widget
# 2. Click "Add Plane" multiple times
# 3. Configure each plane independently
# 4. The scene will be clipped by all planes
```

---

## 🐛 Troubleshooting

### Common Issues

**Issue: "ModuleNotFoundError: No module named 'evsplitting'"**
```bash
# Solution: Build CUDA extension again
cd gaussian-splatting/submodules/ev-splitting
pip install -e .
cd ../../..
```

**Issue: CUDA out of memory**
```bash
# Solutions:
# 1. Use Scene Graph or CPU Offload memory mode
# 2. Reduce EVS passes (evs_max_passes = 1)
# 3. Reduce image resolution
# 4. Use a smaller scene
```

**Issue: Slow rendering performance**
```bash
# Solutions:
# 1. Start with evs_max_passes = 1
# 2. Disable Debug Render (if enabled)
# 3. Disable memory measurement (if enabled)
# 4. Use Naive memory mode for speed
```

**Issue: Clipping plane not visible**
```bash
# Solutions:
# 1. Check "Visualize Plane" in Clipping Plane widget
# 2. Ensure plane intersects with Gaussians
# 3. Check plane normal vector (should be normalized)
```

---

## 📝 Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{evs_placeholder_2024,
  title={[Paper Title (Placeholder)]},
  author={[Author(s) (Placeholder)]},
  journal={ACM Transactions on Graphics (TOG)},
  volume={43},
  number={6},
  year={2024},
  publisher={ACM},
  doi={10.1145/3680528.3687592}
}
```

Also cite the related works:

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

## 🙏 Acknowledgements

This project stands on the shoulders of excellent open-source work:

### Primary References

- **[Splatviz](https://github.com/Florian-Barthel/splatviz)** by Florian Barthel
  - Interactive 3D Gaussian Splatting viewer
  - GUI framework and rendering pipeline

- **[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)** by INRIA GRAPHDECO
  - Original 3D Gaussian Splatting implementation
  - Core Gaussian representation and rasterization

- **[diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)**
  - Differentiable Gaussian rasterization
  - CUDA-based rendering backend

### Technologies Used

- **PyTorch** - Deep learning framework
- **CUDA/cuDNN** - GPU acceleration
- **ImGui** - Immediate mode GUI
- **OpenGL** - Graphics rendering
- **GLM** - Mathematics library

### Special Thanks

Special thanks to the authors of the original EVS paper for their pioneering research and the community for their continued support in 3D Gaussian Splatting research.

---

## 📄 License

This project is licensed for **non-commercial research and evaluation use only**, following the terms of the INRIA Gaussian Splatting license.

### Permissions

✅ **Allowed:**
- Academic research and publication
- Educational use
- Non-commercial evaluation and comparison
- Derivative research works (with attribution)

❌ **Not Allowed:**
- Commercial applications
- Commercial licensing or resale
- Incorporation into proprietary products without explicit permission

### Commercial Use

For commercial licensing inquiries, please contact the original paper authors.

---

## 🐛 Issues and Contributions

### Reporting Issues

- Please use GitHub Issues for bug reports
- Include:
  - Error message and traceback
  - System information (OS, GPU, CUDA version, Python version)
  - Steps to reproduce
  - Screenshots or logs if applicable

### Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Submit a pull request with a clear description

### Research Questions

For questions about the original EVS algorithm or paper:
- Refer to the original paper at [10.1145/3680528.3687592](https://dl.acm.org/doi/full/10.1145/3680528.3687592)
- Contact the original paper authors

---

## 📧 Contact & Support

- **Implementation Issues:** Open a GitHub Issue
- **Feature Requests:** GitHub Discussions or Issues
- **Research Collaboration:** Contact authors of the original paper
- **Splatviz Questions:** See [Splatviz Repository](https://github.com/Florian-Barthel/splatviz)

---

## 📖 Additional Resources

- **3D Gaussian Splatting Paper:** [Kerbl et al., 2023](https://repo.cvg.ethz.ch/projects/3dgs/)
- **Splatviz Documentation:** [GitHub Wiki](https://github.com/Florian-Barthel/splatviz/wiki)
- **CUDA Programming Guide:** [NVIDIA Docs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

---

<p align="center">
  <strong>Made with ❤️ for the 3D Gaussian Splatting community</strong>
  <br>
  <br>
  <a href="https://github.com">⭐ Star this repository if you find it useful!</a>
</p>
