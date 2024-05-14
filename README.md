# JABAS: Joint Adaptive Batching and Automatic Scaling for DNN Training on Heterogeneous GPUs
## Introduction
*JABAS* (*J*oint *A*daptive *B*atching and *A*utomatic *S*caling) is a novel DNN training system for a heterogeneous GPU cluster.
Major components of JABAS are a DNN training framework called IIDP,
which provides the same theoretical convergence rate of distributed SGD
in a heterogeneous GPU cluster,
a fine-grained adaptive batching technique with dynamic configuration,
and a coarse-grained automatic resource scaling technique that leverages the prediction of global batch size changes for an epoch to auto-scale GPU resources optimally.

## Getting Started
### Prerequisites
* Anaconda3 (Python3.6)
* CUDA 11.1
* cuDNN 8.2.1

### Build conda environment
```bash
conda create -n jabas python=3.6
conda activate jabas
conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
conda install -c pytorch magma-cuda111 # For cuda11.1

git clone --recursive https://github.com/eurosys25-anonymous-authors/JABAS.git
cd JABAS/
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py install

# Install torchvision
cd vision/
python setup.py install
```
