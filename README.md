# FastLoop: Fast Loop Closure for Visual SLAM

## Overview
FastLoop is a high-performance loop closure framework designed for visual SLAM systems.  
The goal of this project is to accelerate loop detection and correction while maintaining accurate trajectory estimation.

This implementation focuses on:
- Efficient loop candidate detection
- Fast pose graph optimization
- Parallelizable components for improved scalability

The system is designed to integrate with modern SLAM pipelines and can be extended for GPU acceleration.

---

## Features

- Fast loop closure detection
- Pose graph optimization
- Parallelizable architecture
- Research-oriented implementation

---

## Prerequisites

The library has been tested on **Ubuntu 20.04** and **Ubuntu 22.04**. It should also compile on other platforms with minimal adjustments. For real-time performance and more stable results, we recommend running the system on a relatively powerful machine (e.g., an Intel i7 or similar).

## C++14 Compiler

A compiler supporting the **C++14** standard is required to build and run TurboMap.

## CUDA

The library has been tested with **CUDA 12.5**. Installation instructions are available in the official NVIDIA guide:  
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/

## OpenCV

We use **OpenCV** for image processing and feature manipulation.  
Installation instructions can be found at: http://opencv.org  

**Minimum required version: 4.4.0**

## Eigen3

**Eigen3** is required by the optimization library g2o (see below).  
Download and installation instructions are available at: http://eigen.tuxfamily.org  

**Minimum required version: 3.1.0**

## Pangolin

**Pangolin** is used for visualization and the graphical user interface.  
Installation instructions are available here:  
https://github.com/stevenlovegrove/Pangolin

Because Pangolin avoids using Eigen in CUDA, compiling this project requires commenting out the guards in:

- line **475** of `glsl.hpp`
- line **47** of `glsl.h`

More details can be found in this issue:  
https://github.com/stevenlovegrove/Pangolin/issues/814

## DBoW2 and g2o (Included in the Thirdparty Folder)

We use modified versions of:

- **DBoW2** for place recognition  
- **g2o** for nonlinear optimization

Both libraries are included in the `Thirdparty` folder and are distributed under the **BSD license**.

## Python

Python is required for aligning the estimated trajectory with the ground truth.  

**Required dependency:**  
- `NumPy`
  
Installation options:

- **Windows:** http://www.python.org/downloads/windows  
- **Ubuntu/Debian:**  
  ```bash
  sudo apt install libpython2.7-dev
  
---

## Build

## Clone the Repository
Start by cloning the project repository:

```bash
git clone [git@github.com:sfu-rsl/FastLoop.git](https://github.com/sfu-rsl/FastLoop.git)
```
## Install GPU Block Solver
Once the repository is cloned, follow the instructions provided [here](https://github.com/sfu-rsl/gpu-block-solver/blob/master/INTEGRATION.md) to install the accelerated GPU block solver.

## Build the Project
This project is built on top of ORB-SLAM3, which includes a script called `build.sh` to compile both the Thirdparty libraries and ORB-SLAM3 itself.

Then run the following commands:

```
cd FastLoop
chmod +x build.sh
./build.sh
```
## Output
After a successful build:

- The shared library libORB_SLAM3.so will be generated in the lib directory
- Executable files will be available in the Examples directory
