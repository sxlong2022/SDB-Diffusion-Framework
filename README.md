# BathySurrogate

An Open-Source Environmental Surrogate Framework for Satellite-Derived Bathymetry via Multi-Source Data Fusion and Spatial Validation.

## Overview

**BathySurrogate** is an open-source, modular Python computational framework designed for high-resolution Satellite-Derived Bathymetry (SDB) mapping in turbid and complex coastal waters. It implements a multi-source data fusion surrogate paradigm--coupling multi-temporal Sentinel-2 optical composites, GEBCO global bathymetric priors, and sparse in-situ nautical chart soundings--evaluated through a leak-proof Spatially Blocked 5-Fold Cross-Validation protocol.

The framework features a 12-feature Enhanced Random Forest surrogate engine as its primary operational predictor, alongside an inverse score-based generative diffusion model (S3GM) with guided Diffusion Posterior Sampling (DPS) for conditional spatio-temporal reconstruction and uncertainty analysis.

## Highlights

- **Modular Software Design**: Implements Strategy, Factory, and Pipeline architectural design patterns with externalized YAML configuration management.
- **Leak-Proof Spatial Validation**: Enforces Spatially Blocked 5-Fold Cross-Validation ($4 \times 4$ spatial grid) ensuring zero spatial autocorrelation leakage between training and evaluation splits.
- **Enhanced Surrogate Modeling**: Integrates 12 domain-specific features (spectral bands, band ratios, log transforms, spatial coordinates, GEBCO bathymetric priors, and nearest-neighbor sounding distances/depths) achieving pooled Out-of-Fold (OOF) $R^2 = 0.483$, Pearson $r = 0.706$, $\text{RMSE} = 9.19\text{ m}$, $\text{MAE} = 6.61\text{ m}$, and a range-normalized $\text{nRMSE}_{\text{range}} = 12.61\%$.
- **High-Performance Computation**: Low computational complexity ($O(S \times N \times C^2)$), processing annual decadal mapping workflows in seconds on standard CPUs and minutes on consumer GPUs (NVIDIA RTX 2060, 6GB VRAM).
- **FAIR4RS Compliance**: Adheres strictly to Findable, Accessible, Interoperable, and Reusable software standards with comprehensive test suites, metadata (`CITATION.cff`), and pinned environment definitions (`environment.yml`).

## System Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support and >=6GB VRAM (tested on NVIDIA GeForce RTX 2060 / RTX 4060 Laptop GPU)
- **RAM**: >=16GB system memory
- **Storage**: >=50GB for raw imagery, intermediate arrays, and cartographic products

### Software Requirements
- **Operating System**: Linux (Ubuntu 20.04+), Windows (10/11), or macOS
- **Python**: 3.9+
- **CUDA**: >=11.8 (tested on CUDA 12.4)
- **Conda**: Recommended for reproducible environment management

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/sxlong2022/BathySurrogate.git
cd BathySurrogate
```

### Step 2: Create Conda Environment

```bash
conda env create -f environment.yml
conda activate bathysurrogate
```

The `environment.yml` includes all required dependencies:
- PyTorch >=2.0.1 (with CUDA acceleration)
- scikit-learn >=1.6
- NumPy, SciPy
- Google Earth Engine Python API (`earthengine-api`)
- netCDF4, Matplotlib, PyYAML, Pillow, Joblib, tqdm

### Step 3: Verify Installation

Run the automated test suite:

```bash
python tests/quick_test.py
```

All verification checks should report `[PASS]`.

## Project Structure

```
|-- run_bathymetry.py                  # Main CLI entry point for all execution stages
|-- data_acquisition_preprocessing.py      # Sentinel-2 & GEBCO data acquisition (GEE)
|-- miwc.py                            # Multi-temporal Image Weighted Composition (MIWC)
|-- environment.yml                    # Conda environment definition
|-- CITATION.cff                       # Citation metadata format
|-- LICENSE                            # MIT License
|-- tests/
|   \-- quick_test.py                  # Automated test suite and configuration validator
|-- bathysurrogate/                    # Core BathySurrogate Python package
|   |-- __init__.py                    # Package initialization and public API
|   |-- main.py                        # Hybrid system orchestrator (HybridBathymetrySystem)
|   |-- classic_models.py              # Classic & 12-feature Enhanced Random Forest surrogate
|   |-- preprocessor.py                # Min-Max normalization & spatial blocking partitions
|   |-- s3gm_wrapper.py                # S3GM generative diffusion model wrapper
|   |-- s3gm_config.py                 # S3GM hyperparameter dataclass
|   |-- gpu_memory.py                  # GPU VRAM garbage collection and monitoring
|   \-- utils.py                       # Spatio-temporal array utilities and validation
|-- configs/                           # Externalized YAML configuration files
|   |-- classic_models.yaml            # Random Forest hyperparameters & feature configs
|   \-- s3gm_default.yaml              # Diffusion SDE, VPSDE, and DPS sampling parameters
\-- S3GM/Code/                         # S3GM neural core (adapted from Li et al., 2024)
    |-- models/                        # U-Net video architecture with RPE
    |-- sampler/                       # VP-SDE predictor-corrector with DPS guidance
    \-- trainer/                       # Score matching loss and dataset loaders
```

## Quick Start

### 1. Running the Test Suite

```bash
python tests/quick_test.py
```

This verifies:
1. Core dependency imports (NumPy, PyYAML, PyTorch, scikit-learn, `bathysurrogate`)
2. CUDA GPU availability and memory detection
3. Synthetic Random Forest surrogate training and inference
4. YAML configuration loading and parameter validation
5. S3GM model wrapper configuration initialization

### 2. Multi-Stage Pipeline Execution

The framework operates via modular CLI stages:

```bash
# Stage 1: Data Acquisition & Preprocessing (Sentinel-2 MIWC + GEBCO downsampling)
python run_bathymetry.py --stage 1

# Stage 1.5: Surrogate Model Training (Full & 5-Fold Spatially Blocked CV)
python run_bathymetry.py --stage 1.5

# Stage 1.8: Surrogate Model Out-of-Fold (OOF) Spatial Validation
python run_bathymetry.py --stage 1.8

# Stage 2: S3GM Spatio-Temporal Generative Diffusion Sampling
python run_bathymetry.py --stage 2

# Stage 3: Post-processing, Denormalization, and Spatial Visualization
python run_bathymetry.py --stage 3

# Stage 4: Statistical Significance Analysis (Wilcoxon Signed-Rank Test)
python run_bathymetry.py --stage 4

# Stage 5: Detailed Performance Stratification Across Depth & Slope Zones
python run_bathymetry.py --stage 5
```

> **Note on Google Earth Engine**: Before running Stage 1, ensure you authenticate and initialize your GEE account:
> ```python
> ee.Authenticate()
> ee.Initialize(project='YOUR_GEE_PROJECT_ID')
> ```

## Configuration

All model hyperparameters and feature engineering settings are externalized into human-readable YAML files:

### Random Forest Configuration (`configs/classic_models.yaml`)

```yaml
rf_params:
  model_params:
    n_estimators: 500
    max_depth: 25
    max_features: 'sqrt'
    random_state: 42
    n_jobs: -1
    oob_score: true
```

### S3GM Diffusion Configuration (`configs/s3gm_default.yaml`)

```yaml
sde_type: 'vpsde'
beta_min: 0.1
beta_max: 1000  # VPSDE scaling parameter

sampling:
  alpha: 0.1             # DPS data fidelity guidance weight
  gamma_spatial: 0.1     # Spatial smoothness regularization weight
  inner_loop: 1          # Langevin corrector steps
  snr: 0.01              # Signal-to-noise ratio
```

## Citation

If you use BathySurrogate in your research, please cite:

```bibtex
@article{song2026bathysurrogate,
  title={BathySurrogate: An Open-Source Environmental Surrogate Framework for Satellite-Derived Bathymetry via Multi-Source Data Fusion and Spatial Validation},
  author={Song, Xiaolong and Liu, Boliang and Xiao, Zhong and Xu, Haijue and Bai, Yuchuan},
  journal={Environmental Modelling \& Software},
  year={2026},
  note={Under Review}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact & Issues

- **GitHub Issues**: https://github.com/sxlong2022/BathySurrogate/issues
- **Maintainer**: Xiaolong Song (xlsong@tju.edu.cn), Tianjin University
