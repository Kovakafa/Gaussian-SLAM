<p align="center">
  <h1 align="center">Gaussian-SLAM on Jetson Orin: TUM RGB-D Dataset Experiments</h1>
  <p align="center">
    <strong>Hüseyin Mert Çalışkan</strong>
  </p>
  <div align="center"><small>Graduation Project - Based on <a href="https://github.com/VladimirYugay/Gaussian-SLAM">Gaussian-SLAM</a> by Vladimir Yugay et al.</small></div>
  <h3 align="center"><a href="https://github.com/Kovakafa/Gaussian-SLAM/blob/main/docs/GS-SLAM_JetsonOrin.pdf">📄 Research Paper</a></h3>
  <div align="center"></div>
</p>
<p align="center">
  <a href="">
    <img src="./assets/jetson_results.gif" width="90%">
  </a>
</p>

## 🚀 Overview
This repository contains an adaptation of Gaussian-SLAM optimized for **NVIDIA Jetson Orin** platform with comprehensive experiments on the **TUM RGB-D dataset**. This graduation project includes memory optimizations, CUDA 12.2 compatibility, and performance benchmarks for real-time inference on edge computing hardware.

## ⚙️ Setting Things Up
Clone the repo:
```bash
git clone https://github.com/Kovakafa/Gaussian-SLAM.git
cd Gaussian-SLAM
```

Make sure that gcc and g++ paths on your system are exported:
```bash
export CC=<gcc path>
export CXX=<g++ path>
```

### Environment Setup for Jetson Orin with CUDA 12.2
**Important**: The original environment file has been modified for CUDA 12.2 compatibility and Jetson Orin optimization.

```bash
# Create environment from modified conda file
conda env create -f environment_jetson.yml
conda activate gslam-jetson
```

### Key Environment Modifications
- **CUDA**: 12.2 (instead of original version)
- **PyTorch**: 2.1.0 compatible with CUDA 12.2
- **Torchvision**: 0.18.0 (optimized for Jetson Orin)
- **Memory optimizations**: Reduced package versions for Jetson constraints

```bash
# Install additional Jetson-specific dependencies
pip install -r requirements_jetson.txt
```

We tested our code on **Jetson Orin NX 16GB** with **Ubuntu 20.04** and **CUDA 12.2**.

## 🔨 Running Gaussian-SLAM on Jetson Orin

  <details>
  <summary><b>Downloading TUM RGB-D Dataset</b></summary>
  
  Download the TUM RGB-D dataset:
  ```bash
  # Create datasets directory
  mkdir datasets && cd datasets
  
  # Download TUM RGB-D sequences
  wget https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz
  wget https://vision.in.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_xyz.tgz
  wget https://vision.in.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz
  
  # Extract datasets
  tar -xzf rgbd_dataset_freiburg1_xyz.tgz
  tar -xzf rgbd_dataset_freiburg2_xyz.tgz
  tar -xzf rgbd_dataset_freiburg3_long_office_household.tgz
  ```
  
  Or download from the official TUM RGB-D Dataset page: https://vision.in.tum.de/data/datasets/rgbd-dataset
  
  The dataset should be organized as:
  ```
  datasets/
  ├── rgbd_dataset_freiburg1_xyz/
  ├── rgbd_dataset_freiburg2_xyz/
  └── rgbd_dataset_freiburg3_long_office_household/
  ```
  
  </details>

  <details>
  <summary><b>Running the code</b></summary>
  
  Start the system with the command:
  ```bash
  python run_slam.py configs/TUM_RGBD/<config_name> --input_path <path_to_scene> --output_path <output_path>
  ```
  
  For example:
  ```bash
  python run_slam.py configs/TUM_RGBD/freiburg1_xyz.yaml --input_path datasets/rgbd_dataset_freiburg1_xyz --output_path output/TUM_RGBD/freiburg1_xyz
  ```
  
  For real-time inference on Jetson Orin:
  ```bash
  python real_time_slam.py --camera_id 0 --config configs/jetson_real_time.yaml
  ```
  
  </details>

  <details>
  <summary><b>Jetson Orin Optimizations</b></summary>
  
  Key optimizations implemented for Jetson Orin with CUDA 12.2:
  
  **Environment Optimizations:**
  - Modified conda environment for CUDA 12.2 compatibility
  - Torchvision 0.18.0 for improved Jetson performance
  - Memory-efficient package versions
  
  **Runtime Optimizations:**
  - **Memory Management**: Adaptive Gaussian pruning for 16GB memory constraint
  - **Mixed Precision**: FP16 inference with CUDA 12.2 optimizations
  - **Batch Processing**: Optimized batch sizes for Jetson Orin architecture
  - **Real-time Pipeline**: Asynchronous processing and adaptive frame rate control
  
  </details>

  <details>
  <summary><b>Performance Results on TUM RGB-D</b></summary>
  
  Performance metrics on Jetson Orin NX 16GB with TUM RGB-D dataset:
  
  | Sequence | RMSE (cm) | FPS | Memory Usage (GB) | Power (W) |
  |----------|-----------|-----|-------------------|-----------|
  | freiburg1_xyz | 2.14 | 12.3 | 14.2 | 22.5 |
  | freiburg2_xyz | 1.89 | 11.8 | 13.8 | 23.1 |
  | freiburg3_long | 2.31 | 10.2 | 15.1 | 24.3 |
  
  **Comparison with Desktop GPU:**
  - Jetson Orin NX: 11.4 fps avg, 23.3 W power consumption
  - RTX 4090: 38.7 fps avg, 450+ W power consumption
  - Power efficiency: ~30% of desktop performance at ~5% power consumption
  
  Detailed results and analysis available in our [research paper](https://github.com/Kovakafa/Gaussian-SLAM/blob/main/docs/GS-SLAM_JetsonOrin.pdf).
  
  </details>

  <details>
  <summary><b>Environment File Modifications</b></summary>
  
  The original `environment.yml` has been modified for Jetson Orin compatibility:
  
  **Changed Dependencies:**
  ```yaml
  # Original vs Modified
  pytorch: 1.13.0 → 2.1.0+cu122
  torchvision: 0.14.0 → 0.18.0
  cudatoolkit: 11.7 → 12.2
  ```
  
  **Added Jetson-specific packages:**
  ```yaml
  - jetson-stats
  - jtop
  - tensorrt  # for inference optimization
  ```
  
  View complete environment file: [environment_jetson.yml](environment_jetson.yml)
  
  </details>

## 📄 Research Paper
Our detailed research paper covering the Jetson Orin adaptation, optimization strategies, and TUM RGB-D experiments:

**📋 [GS-SLAM_JetsonOrin.pdf](https://github.com/Kovakafa/Gaussian-SLAM/blob/main/docs/GS-SLAM_JetsonOrin.pdf)**

## 📌 Citation
If you find our adaptation useful, please cite both the original work and our contribution:

**Original Gaussian-SLAM:**
```bib
@misc{yugay2023gaussianslam,
      title={Gaussian-SLAM: Photo-realistic Dense SLAM with Gaussian Splatting}, 
      author={Vladimir Yugay and Yue Li and Theo Gevers and Martin R. Oswald},
      year={2023},
      eprint={2312.10070},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

**This work:**
```bib
@misc{caliskan2024gaussianslam,
      title={Gaussian-SLAM on Jetson Orin: Real-time Dense SLAM for Edge Computing with TUM RGB-D Dataset}, 
      author={Hüseyin Mert Çalışkan},
      year={2024},
      note={Graduation Project - Available at: https://github.com/Kovakafa/Gaussian-SLAM}
}
```
