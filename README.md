# grad_routing

### High-performance, differentiable, Cython-accelerated flow-routing toolkit (PCRaster-style LDD)  
**Author:** Dinesh Joshi  
**License:** Licensed under the Apache License 2.0.

Citation
Joshi, D. (2025). Grad_Routing: Differentiable Flow Accumulation Python Package for Hydrological Modeling (0.1.0). Zenodo. https://doi.org/10.5281/zenodo.17763733

---

## Overview

`grad_routing` is a high-performance flow-routing toolkit designed for hydrology, environmental modeling, and ML-based hydrological simulations.  
The core routing kernels are compiled using **Cython** into fast `.pyd` or `.so` binaries, while TensorFlow-compatible functions allow differentiable routing for machine-learning workflows.

The package supports **PCRaster-style Local Drain Direction (LDD)** grids and provides fast utilities for:

- Topological tier generation  
- Differentiable flow accumulation  
- Acyclic flow routing networks  
- GPU/XLA-optimized operations  

---

## Features

- **Cython-accelerated** routing kernels  
- **Differentiable** functions compatible with TensorFlow autograd  
- **PCRaster LDD (1–9) support**  
  - Directional codes: 1–4, 6–9  
  - `5` = outlet / no-flow  
- **XLA/GPU-friendly** algorithms  
- Designed for **hydrology, ML routing layers, and DAG simulations**

---

## Installation

### Install using Wheel

Pre-built wheel files for `grad_routing` are provided for different Python versions and platforms:

- **Windows:** Python 3.11  
- **Linux:** Python 3.12  

To install the package from the wheel file, navigate to the folder containing the appropriate `.whl` file and run:

```bash
# For Windows (Python 3.11)
pip install grad_routing-0.1.0-cp311-cp311-win_amd64.whl

# For Linux (Python 3.12)
pip install grad_routing-0.1.0-cp312-cp312-linux_x86_64.whl

### Install using pip (from PyPI)

Once published on PyPI, the package can be installed directly using:

```bash
pip install grad_routing

