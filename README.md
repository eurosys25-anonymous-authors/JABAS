# JABAS: Joint Adaptive Batching and Automatic Scaling for DNN Training on Heterogeneous GPUs
## Introduction
*JABAS* (*J*oint *A*daptive *B*atching and *A*utomatic *S*caling) is a novel DNN training system for a heterogeneous GPU cluster.
Major components of JABAS are a DNN training framework called IIDP, which provides the same theoretical convergence rate of distributed SGD in a heterogeneous GPU cluster,
a fine-grained adaptive batching technique with dynamic configuration,
and a coarse-grained automatic resource scaling technique that leverages the prediction of global batch size changes for an epoch to auto-scale GPU resources optimally.

## Getting Started
### Prerequisites
* Anaconda3 (Python3.6)
* CUDA 11.1
* cuDNN 8.2.1

### Build conda environment
```bash
conda create -n jabas python=3.6 -y
conda activate jabas
conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses -y
conda install -c pytorch magma-cuda111 -y # For CUDA 11.1

git clone --recursive -b v1.8.1 https://github.com/pytorch/pytorch.git
cd pytorch/
patch -p1 < ../pytorch.patch
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py install
pip install -r requirements.txt

# Install torchvision
git clone -b v0.9.1 https://github.com/pytorch/vision.git
cd vision/
python setup.py install
```
